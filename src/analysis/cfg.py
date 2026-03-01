"""
Control flow graph construction and analysis for the Cross-Language
Equivalence Verifier.

Provides: CFG construction from basic blocks, dominator trees
(Lengauer-Tarjan), post-dominator trees, dominance frontiers, natural
loop detection (SCC-based), loop nesting trees, unreachable block
detection, critical edge splitting, and back edge identification.
"""

from __future__ import annotations

from collections import deque
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
from ..ir.instructions import (
    BranchInst,
    SwitchInst,
    ReturnInst,
    Instruction,
    is_terminator,
    get_successors,
)


# ── Edge classification ─────────────────────────────────────────────────

class EdgeKind(Enum):
    """Classification of a CFG edge."""
    TREE = auto()       # Tree edge (part of DFS tree)
    FORWARD = auto()    # Forward edge (ancestor → descendant, not tree)
    BACK = auto()       # Back edge (descendant → ancestor / self-loop)
    CROSS = auto()      # Cross edge (between unrelated subtrees)

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class CFGEdge:
    """A directed edge in the CFG."""
    src: BasicBlock
    dst: BasicBlock
    kind: EdgeKind = EdgeKind.TREE
    is_critical: bool = False  # True if src has >1 successors AND dst has >1 predecessors

    def __str__(self) -> str:
        crit = " [critical]" if self.is_critical else ""
        return f"{self.src.name} → {self.dst.name} ({self.kind}){crit}"


# ── CFG ──────────────────────────────────────────────────────────────────

class CFG:
    """Control flow graph for a single function.

    Wraps the basic-block predecessor/successor lists with higher-level
    analysis: edge classification, dominator trees, loop detection, etc.
    """

    def __init__(self, function: Function) -> None:
        self.function = function
        self._blocks: list[BasicBlock] = function.blocks
        self._entry: BasicBlock | None = function.entry_block
        self._edges: list[CFGEdge] = []

        # Analysis caches
        self._dom_tree: DominatorTree | None = None
        self._post_dom_tree: DominatorTree | None = None
        self._loop_info: LoopInfo | None = None
        self._dfs_order: list[BasicBlock] = []
        self._rpo_order: list[BasicBlock] = []
        self._edge_kinds: dict[Tuple[int, int], EdgeKind] = {}

        self._build()

    # ── Construction ─────────────────────────────────────────────────────

    def _build(self) -> None:
        """Build the CFG edge list and classify edges."""
        if not self._blocks or not self._entry:
            return

        # Collect edges
        for block in self._blocks:
            for succ in block.successors:
                is_crit = len(block.successors) > 1 and len(succ.predecessors) > 1
                self._edges.append(CFGEdge(
                    src=block, dst=succ, is_critical=is_crit,
                ))

        # DFS for edge classification and ordering
        self._classify_edges()

    def _classify_edges(self) -> None:
        """Classify edges via DFS and compute DFS/RPO orderings."""
        if not self._entry:
            return

        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[int, int] = {b.id: WHITE for b in self._blocks}
        dfs_num: dict[int, int] = {}
        finish_num: dict[int, int] = {}
        counter = [0]
        rpo: list[BasicBlock] = []

        def dfs(block: BasicBlock) -> None:
            color[block.id] = GRAY
            dfs_num[block.id] = counter[0]
            counter[0] += 1
            self._dfs_order.append(block)

            for succ in block.successors:
                key = (block.id, succ.id)
                if color[succ.id] == WHITE:
                    self._edge_kinds[key] = EdgeKind.TREE
                    dfs(succ)
                elif color[succ.id] == GRAY:
                    self._edge_kinds[key] = EdgeKind.BACK
                else:  # BLACK
                    if dfs_num[block.id] < dfs_num[succ.id]:
                        self._edge_kinds[key] = EdgeKind.FORWARD
                    else:
                        self._edge_kinds[key] = EdgeKind.CROSS

            color[block.id] = BLACK
            finish_num[block.id] = counter[0]
            counter[0] += 1
            rpo.append(block)

        dfs(self._entry)
        rpo.reverse()
        self._rpo_order = rpo

        # Update edge objects with kinds
        for i, edge in enumerate(self._edges):
            key = (edge.src.id, edge.dst.id)
            if key in self._edge_kinds:
                self._edges[i] = CFGEdge(
                    src=edge.src, dst=edge.dst,
                    kind=self._edge_kinds[key],
                    is_critical=edge.is_critical,
                )

    # ── Accessors ────────────────────────────────────────────────────────

    @property
    def entry(self) -> BasicBlock | None:
        return self._entry

    @property
    def blocks(self) -> list[BasicBlock]:
        return list(self._blocks)

    @property
    def num_blocks(self) -> int:
        return len(self._blocks)

    @property
    def edges(self) -> list[CFGEdge]:
        return list(self._edges)

    @property
    def num_edges(self) -> int:
        return len(self._edges)

    def edge_kind(self, src: BasicBlock, dst: BasicBlock) -> EdgeKind | None:
        return self._edge_kinds.get((src.id, dst.id))

    @property
    def back_edges(self) -> list[CFGEdge]:
        return [e for e in self._edges if e.kind is EdgeKind.BACK]

    @property
    def critical_edges(self) -> list[CFGEdge]:
        return [e for e in self._edges if e.is_critical]

    @property
    def dfs_order(self) -> list[BasicBlock]:
        return list(self._dfs_order)

    @property
    def rpo_order(self) -> list[BasicBlock]:
        return list(self._rpo_order)

    # ── Reachability ─────────────────────────────────────────────────────

    def reachable_blocks(self) -> set[BasicBlock]:
        """Return the set of blocks reachable from the entry."""
        if not self._entry:
            return set()
        visited: set[int] = set()
        queue: deque[BasicBlock] = deque([self._entry])
        visited.add(self._entry.id)
        result: set[BasicBlock] = {self._entry}
        while queue:
            block = queue.popleft()
            for succ in block.successors:
                if succ.id not in visited:
                    visited.add(succ.id)
                    result.add(succ)
                    queue.append(succ)
        return result

    def unreachable_blocks(self) -> set[BasicBlock]:
        """Return blocks not reachable from the entry."""
        reachable = self.reachable_blocks()
        return set(self._blocks) - reachable

    def is_reachable(self, block: BasicBlock) -> bool:
        return block in self.reachable_blocks()

    # ── Dominator tree ───────────────────────────────────────────────────

    @property
    def dominator_tree(self) -> "DominatorTree":
        if self._dom_tree is None:
            self._dom_tree = DominatorTree.build(self, post=False)
        return self._dom_tree

    @property
    def post_dominator_tree(self) -> "DominatorTree":
        if self._post_dom_tree is None:
            self._post_dom_tree = DominatorTree.build(self, post=True)
        return self._post_dom_tree

    # ── Loop detection ───────────────────────────────────────────────────

    @property
    def loop_info(self) -> "LoopInfo":
        if self._loop_info is None:
            self._loop_info = LoopInfo.build(self)
        return self._loop_info

    # ── Critical edge splitting ──────────────────────────────────────────

    def split_critical_edges(self) -> list[BasicBlock]:
        """Split all critical edges by inserting empty blocks.

        Returns the list of newly created blocks.
        """
        new_blocks: list[BasicBlock] = []
        crits = self.critical_edges

        for edge in crits:
            src, dst = edge.src, edge.dst
            # Create a new block between src and dst
            new_block = BasicBlock(
                name=f"{src.name}_to_{dst.name}",
                parent=self.function,
            )
            # Add unconditional branch to dst
            br = BranchInst(target=dst)
            new_block.append(br)

            # Update src's terminator to point to new_block instead of dst
            self._retarget_edge(src, dst, new_block)

            # Update phi nodes in dst
            for phi in dst.phi_nodes:
                for i, (v, b) in enumerate(phi._incoming):
                    if b is src:
                        phi._incoming[i] = (v, new_block)

            # Insert into function
            if self.function:
                src_idx = self.function._blocks.index(src)
                self.function._blocks.insert(src_idx + 1, new_block)

            new_blocks.append(new_block)

        # Invalidate caches
        if new_blocks:
            self._dom_tree = None
            self._post_dom_tree = None
            self._loop_info = None

        return new_blocks

    def _retarget_edge(
        self, src: BasicBlock, old_dst: BasicBlock, new_dst: BasicBlock,
    ) -> None:
        """Retarget the edge from src→old_dst to src→new_dst."""
        term = src.terminator
        if term is None:
            return

        if isinstance(term, BranchInst):
            if term._true_target is old_dst:
                term._true_target = new_dst
            if term._false_target is old_dst:
                term._false_target = new_dst
        elif isinstance(term, SwitchInst):
            if term._default_target is old_dst:
                term._default_target = new_dst
            term._cases = [
                (c, new_dst if b is old_dst else b) for c, b in term._cases
            ]

        # Update successor lists
        src._successors = [new_dst if s is old_dst else s for s in src._successors]
        old_dst.remove_predecessor(src)
        new_dst.add_predecessor(src)

    # ── Summary ──────────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [
            f"CFG for '{self.function.name}':",
            f"  Blocks:       {self.num_blocks}",
            f"  Edges:        {self.num_edges}",
            f"  Back edges:   {len(self.back_edges)}",
            f"  Critical:     {len(self.critical_edges)}",
            f"  Unreachable:  {len(self.unreachable_blocks())}",
        ]
        return "\n".join(lines)


# ── Dominator Tree ───────────────────────────────────────────────────────

class DominatorTree:
    """Dominator (or post-dominator) tree.

    Built using the Lengauer-Tarjan algorithm, then computes dominance
    frontiers.
    """

    def __init__(self) -> None:
        self._idom: dict[int, BasicBlock | None] = {}
        self._children: dict[int, list[BasicBlock]] = {}
        self._frontier: dict[int, list[BasicBlock]] = {}
        self._depth: dict[int, int] = {}
        self._is_post: bool = False

    # ── Public API ───────────────────────────────────────────────────────

    def idom(self, block: BasicBlock) -> BasicBlock | None:
        return self._idom.get(block.id)

    def children(self, block: BasicBlock) -> list[BasicBlock]:
        return list(self._children.get(block.id, []))

    def frontier(self, block: BasicBlock) -> list[BasicBlock]:
        return list(self._frontier.get(block.id, []))

    def depth(self, block: BasicBlock) -> int:
        return self._depth.get(block.id, 0)

    def dominates(self, a: BasicBlock, b: BasicBlock) -> bool:
        """Return True if a dominates b."""
        cursor: BasicBlock | None = b
        while cursor is not None:
            if cursor.id == a.id:
                return True
            cursor = self._idom.get(cursor.id)
        return False

    def strictly_dominates(self, a: BasicBlock, b: BasicBlock) -> bool:
        return a.id != b.id and self.dominates(a, b)

    def lca(self, a: BasicBlock, b: BasicBlock) -> BasicBlock | None:
        """Least common ancestor in the dominator tree."""
        a_ancestors: set[int] = set()
        cursor: BasicBlock | None = a
        while cursor is not None:
            a_ancestors.add(cursor.id)
            cursor = self._idom.get(cursor.id)

        cursor = b
        while cursor is not None:
            if cursor.id in a_ancestors:
                return cursor
            cursor = self._idom.get(cursor.id)
        return None

    def preorder(self, root: BasicBlock | None = None) -> list[BasicBlock]:
        """Return blocks in dominator-tree preorder."""
        if root is None:
            # Find the root (block with no idom)
            for bid, idom in self._idom.items():
                if idom is None:
                    root_candidates = [b for b in self._children if b == bid]
                    # Find the actual block object
                    for bid2, children in self._children.items():
                        if bid2 == bid:
                            # We need the block object; search children's parent
                            break
            # Fallback: use depth 0
            for bid, d in self._depth.items():
                if d == 0:
                    root = self._find_block(bid)
                    break
            if root is None:
                return []

        result: list[BasicBlock] = []
        stack = [root]
        while stack:
            block = stack.pop()
            result.append(block)
            children = self._children.get(block.id, [])
            stack.extend(reversed(children))
        return result

    def _find_block(self, block_id: int) -> BasicBlock | None:
        """Find block object by id (scan children lists)."""
        for bid, children in self._children.items():
            for child in children:
                if child.id == block_id:
                    return child
                if bid == block_id:
                    # The parent might be what we want; can find it from idom
                    pass
        return None

    # ── Construction (Lengauer-Tarjan) ───────────────────────────────────

    @classmethod
    def build(cls, cfg: CFG, post: bool = False) -> "DominatorTree":
        """Build the dominator tree using an iterative algorithm.

        Uses Cooper, Harvey, Kennedy's "A Simple, Fast Dominance Algorithm"
        which runs in O(N²) worst case but is fast in practice.

        If post=True, builds the post-dominator tree instead.
        """
        tree = cls()
        tree._is_post = post
        blocks = cfg.blocks
        if not blocks or cfg.entry is None:
            return tree

        # For post-dominators, reverse the graph
        if post:
            # Reverse CFG: add virtual exit, use predecessors as successors
            exit_blocks = [b for b in blocks if b.is_exit]
            if not exit_blocks:
                return tree
            # Use the first exit block as "entry" for reverse graph
            rpo = _reverse_postorder(blocks, exit_blocks[0], reverse_graph=True)
        else:
            rpo = _reverse_postorder(blocks, cfg.entry, reverse_graph=False)

        if not rpo:
            return tree

        block_to_idx = {b.id: i for i, b in enumerate(rpo)}
        n = len(rpo)
        entry_idx = 0

        # idoms array: idoms[i] = index of immediate dominator of rpo[i]
        idoms = [-1] * n
        idoms[entry_idx] = entry_idx

        def _predecessors(b: BasicBlock) -> list[BasicBlock]:
            if post:
                return b.successors
            return b.predecessors

        def intersect(b1: int, b2: int) -> int:
            finger1, finger2 = b1, b2
            while finger1 != finger2:
                while finger1 > finger2:
                    finger1 = idoms[finger1]
                while finger2 > finger1:
                    finger2 = idoms[finger2]
            return finger1

        changed = True
        while changed:
            changed = False
            for i in range(1, n):
                block = rpo[i]
                new_idom = -1
                for pred in _predecessors(block):
                    pred_idx = block_to_idx.get(pred.id, -1)
                    if pred_idx == -1 or idoms[pred_idx] == -1:
                        continue
                    if new_idom == -1:
                        new_idom = pred_idx
                    else:
                        new_idom = intersect(new_idom, pred_idx)
                if new_idom != -1 and idoms[i] != new_idom:
                    idoms[i] = new_idom
                    changed = True

        # Populate tree data structures
        for i, block in enumerate(rpo):
            tree._children[block.id] = []

        for i, block in enumerate(rpo):
            if idoms[i] == i:
                tree._idom[block.id] = None
                tree._depth[block.id] = 0
            else:
                idom_block = rpo[idoms[i]]
                tree._idom[block.id] = idom_block
                tree._children.setdefault(idom_block.id, []).append(block)

        # Compute depths
        for block in rpo:
            _depth = 0
            cursor = tree._idom.get(block.id)
            while cursor is not None:
                _depth += 1
                cursor = tree._idom.get(cursor.id)
            tree._depth[block.id] = _depth

        # Compute dominance frontiers
        for block in rpo:
            tree._frontier[block.id] = []

        for block in rpo:
            preds = _predecessors(block)
            if len(preds) < 2:
                continue
            for pred in preds:
                runner = pred
                idom_block = tree._idom.get(block.id)
                while runner is not None and (idom_block is None or runner.id != idom_block.id):
                    if block not in tree._frontier.get(runner.id, []):
                        tree._frontier.setdefault(runner.id, []).append(block)
                    runner = tree._idom.get(runner.id)

        return tree

    def summary(self) -> str:
        kind = "Post-dominator" if self._is_post else "Dominator"
        lines = [f"{kind} tree:"]
        for bid, idom in sorted(self._idom.items()):
            idom_name = idom.name if idom else "(root)"
            depth = self._depth.get(bid, 0)
            indent = "  " * (depth + 1)
            frontier = [b.name for b in self._frontier.get(bid, [])]
            f_str = f"  DF={frontier}" if frontier else ""
            lines.append(f"{indent}block_{bid} idom={idom_name}{f_str}")
        return "\n".join(lines)


# ── Natural Loop ─────────────────────────────────────────────────────────

@dataclass
class NaturalLoop:
    """A natural loop identified by a back edge."""
    header: BasicBlock
    back_edge_src: BasicBlock
    body: set[BasicBlock] = field(default_factory=set)
    exits: set[BasicBlock] = field(default_factory=set)
    depth: int = 1  # Nesting depth (1 = outermost)
    parent: Optional["NaturalLoop"] = None
    children: list["NaturalLoop"] = field(default_factory=list)

    @property
    def num_blocks(self) -> int:
        return len(self.body)

    @property
    def num_exits(self) -> int:
        return len(self.exits)

    @property
    def is_innermost(self) -> bool:
        return len(self.children) == 0

    def contains(self, block: BasicBlock) -> bool:
        return block in self.body

    def __str__(self) -> str:
        return (
            f"Loop(header={self.header.name}, depth={self.depth}, "
            f"blocks={self.num_blocks}, exits={self.num_exits})"
        )


# ── Loop Info ────────────────────────────────────────────────────────────

class LoopInfo:
    """Loop detection and nesting tree construction.

    Uses back edges from the dominator tree to find natural loops,
    then builds the loop nesting forest.
    """

    def __init__(self) -> None:
        self._loops: list[NaturalLoop] = []
        self._block_to_loop: dict[int, NaturalLoop] = {}
        self._top_level: list[NaturalLoop] = []

    @property
    def loops(self) -> list[NaturalLoop]:
        return list(self._loops)

    @property
    def num_loops(self) -> int:
        return len(self._loops)

    @property
    def top_level_loops(self) -> list[NaturalLoop]:
        return list(self._top_level)

    @property
    def max_depth(self) -> int:
        if not self._loops:
            return 0
        return max(l.depth for l in self._loops)

    def loop_for(self, block: BasicBlock) -> NaturalLoop | None:
        return self._block_to_loop.get(block.id)

    def is_loop_header(self, block: BasicBlock) -> bool:
        return any(l.header.id == block.id for l in self._loops)

    def innermost_loops(self) -> list[NaturalLoop]:
        return [l for l in self._loops if l.is_innermost]

    @classmethod
    def build(cls, cfg: CFG) -> "LoopInfo":
        """Detect natural loops using back edges and dominator tree."""
        info = cls()
        dom = cfg.dominator_tree

        # Find back edges: edge (src, dst) where dst dominates src
        back_edges: list[Tuple[BasicBlock, BasicBlock]] = []
        for edge in cfg.edges:
            if dom.dominates(edge.dst, edge.src):
                back_edges.append((edge.src, edge.dst))

        # For each back edge, compute the natural loop body
        for tail, header in back_edges:
            body = cls._compute_loop_body(header, tail)
            exits = cls._compute_exits(body)
            loop = NaturalLoop(
                header=header,
                back_edge_src=tail,
                body=body,
                exits=exits,
            )
            info._loops.append(loop)

        # Build nesting tree
        # Sort loops by size (larger loops are outer)
        info._loops.sort(key=lambda l: l.num_blocks, reverse=True)

        for i, outer in enumerate(info._loops):
            for inner in info._loops[i + 1:]:
                if inner.header in outer.body and inner.parent is None:
                    inner.parent = outer
                    outer.children.append(inner)

        # Compute depths
        for loop in info._loops:
            depth = 1
            parent = loop.parent
            while parent is not None:
                depth += 1
                parent = parent.parent
            loop.depth = depth

        # Top-level loops
        info._top_level = [l for l in info._loops if l.parent is None]

        # Block-to-loop mapping (innermost loop)
        # Re-sort by size ascending so innermost wins
        for loop in sorted(info._loops, key=lambda l: l.num_blocks):
            for block in loop.body:
                info._block_to_loop[block.id] = loop

        return info

    @staticmethod
    def _compute_loop_body(
        header: BasicBlock, tail: BasicBlock,
    ) -> set[BasicBlock]:
        """Compute the set of blocks in the natural loop for back edge
        tail → header."""
        body: set[BasicBlock] = {header}
        if header is tail:
            return body

        stack: list[BasicBlock] = [tail]
        body.add(tail)
        while stack:
            block = stack.pop()
            for pred in block.predecessors:
                if pred not in body:
                    body.add(pred)
                    stack.append(pred)
        return body

    @staticmethod
    def _compute_exits(body: set[BasicBlock]) -> set[BasicBlock]:
        """Compute exit blocks (successors of body blocks not in body)."""
        exits: set[BasicBlock] = set()
        for block in body:
            for succ in block.successors:
                if succ not in body:
                    exits.add(succ)
        return exits

    def summary(self) -> str:
        lines = [
            f"Loop Info: {self.num_loops} loops, max depth {self.max_depth}",
        ]
        for loop in self._loops:
            indent = "  " * loop.depth
            lines.append(f"{indent}{loop}")
        return "\n".join(lines)


# ── Strongly Connected Components ────────────────────────────────────────

def compute_scc(blocks: list[BasicBlock]) -> list[list[BasicBlock]]:
    """Compute SCCs using Tarjan's algorithm.

    Returns a list of SCCs in reverse topological order.
    """
    index_counter = [0]
    stack: list[BasicBlock] = []
    on_stack: set[int] = set()
    indices: dict[int, int] = {}
    lowlinks: dict[int, int] = {}
    result: list[list[BasicBlock]] = []

    def strongconnect(block: BasicBlock) -> None:
        indices[block.id] = index_counter[0]
        lowlinks[block.id] = index_counter[0]
        index_counter[0] += 1
        stack.append(block)
        on_stack.add(block.id)

        for succ in block.successors:
            if succ.id not in indices:
                strongconnect(succ)
                lowlinks[block.id] = min(lowlinks[block.id], lowlinks[succ.id])
            elif succ.id in on_stack:
                lowlinks[block.id] = min(lowlinks[block.id], indices[succ.id])

        if lowlinks[block.id] == indices[block.id]:
            component: list[BasicBlock] = []
            while True:
                w = stack.pop()
                on_stack.discard(w.id)
                component.append(w)
                if w.id == block.id:
                    break
            result.append(component)

    for block in blocks:
        if block.id not in indices:
            strongconnect(block)

    return result


# ── Helpers ──────────────────────────────────────────────────────────────

def _reverse_postorder(
    blocks: list[BasicBlock],
    entry: BasicBlock,
    reverse_graph: bool = False,
) -> list[BasicBlock]:
    """Compute reverse postorder of blocks from entry."""
    visited: set[int] = set()
    rpo: list[BasicBlock] = []

    def _successors(b: BasicBlock) -> list[BasicBlock]:
        if reverse_graph:
            return b.predecessors
        return b.successors

    def dfs(block: BasicBlock) -> None:
        visited.add(block.id)
        for succ in _successors(block):
            if succ.id not in visited:
                dfs(succ)
        rpo.append(block)

    dfs(entry)
    rpo.reverse()
    return rpo


# ── Convenience ──────────────────────────────────────────────────────────

def build_cfg(function: Function) -> CFG:
    """Build a CFG from a function."""
    return CFG(function)


def find_loops(function: Function) -> LoopInfo:
    """Detect all loops in a function."""
    return CFG(function).loop_info


def find_unreachable(function: Function) -> set[BasicBlock]:
    """Find unreachable blocks in a function."""
    return CFG(function).unreachable_blocks()
