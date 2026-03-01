"""
Call graph construction for the Cross-Language Equivalence Verifier.

Provides: direct and indirect call resolution (using points-to info),
call graph SCC computation, topological ordering, and reachability
analysis.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
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

from ..ir.function import Function
from ..ir.module import Module
from ..ir.instructions import (
    CallInst,
    Instruction,
    Value,
    Constant,
)
from ..ir.types import (
    FunctionType,
    PointerType,
)

from .alias import AndersenAnalysis, PointsToSet


# ── Call site ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CallSite:
    """A single call instruction and its resolved targets."""
    instruction: CallInst
    caller: str                        # Caller function name
    direct_target: str | None = None   # Direct callee name (if known)
    indirect_targets: frozenset[str] = frozenset()  # Possible indirect targets
    is_indirect: bool = False

    @property
    def all_targets(self) -> set[str]:
        targets: set[str] = set()
        if self.direct_target:
            targets.add(self.direct_target)
        targets.update(self.indirect_targets)
        return targets

    def __str__(self) -> str:
        if self.is_indirect:
            targets = ", ".join(sorted(self.all_targets)) or "unknown"
            return f"{self.caller} →? [{targets}]"
        return f"{self.caller} → {self.direct_target}"


# ── Call graph node ──────────────────────────────────────────────────────

@dataclass
class CallGraphNode:
    """A node in the call graph representing a function."""
    name: str
    function: Function | None = None
    callees: set[str] = field(default_factory=set)
    callers: set[str] = field(default_factory=set)
    call_sites: list[CallSite] = field(default_factory=list)
    is_external: bool = False

    @property
    def num_callees(self) -> int:
        return len(self.callees)

    @property
    def num_callers(self) -> int:
        return len(self.callers)

    @property
    def is_leaf(self) -> bool:
        return len(self.callees) == 0

    @property
    def is_root(self) -> bool:
        return len(self.callers) == 0

    @property
    def has_indirect_calls(self) -> bool:
        return any(cs.is_indirect for cs in self.call_sites)

    def __str__(self) -> str:
        return (
            f"CallGraphNode({self.name}, "
            f"callers={self.num_callers}, callees={self.num_callees})"
        )


# ── Call graph SCC ───────────────────────────────────────────────────────

@dataclass
class CallGraphSCC:
    """A strongly connected component in the call graph."""
    functions: frozenset[str]
    is_recursive: bool = False
    nodes: list[CallGraphNode] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.functions)

    @property
    def is_trivial(self) -> bool:
        return self.size == 1 and not self.is_recursive

    def __str__(self) -> str:
        funcs = ", ".join(sorted(self.functions))
        rec = " [recursive]" if self.is_recursive else ""
        return f"SCC({funcs}){rec}"


# ── Call graph ───────────────────────────────────────────────────────────

class CallGraph:
    """Call graph for a module.

    Supports both direct and indirect calls.  Indirect call targets can
    be resolved using points-to analysis results.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, CallGraphNode] = {}
        self._call_sites: list[CallSite] = []
        self._sccs: list[CallGraphSCC] | None = None
        self._topo_order: list[str] | None = None

    # ── Construction ─────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        module: Module,
        points_to: AndersenAnalysis | None = None,
    ) -> "CallGraph":
        """Build the call graph from a module.

        If *points_to* is provided, it is used to resolve indirect calls
        through function pointers.
        """
        cg = cls()

        # Create nodes for all functions
        for func in module.iter_functions():
            node = CallGraphNode(name=func.name, function=func)
            cg._nodes[func.name] = node

        # Create nodes for external declarations
        for ext in module.externals.values():
            if ext.is_function and ext.name not in cg._nodes:
                cg._nodes[ext.name] = CallGraphNode(
                    name=ext.name, is_external=True,
                )

        # Process all call instructions
        for func in module.iter_functions():
            for inst in func.iter_instructions():
                if isinstance(inst, CallInst):
                    call_site = cg._resolve_call(inst, func.name, module, points_to)
                    cg._call_sites.append(call_site)

                    caller_node = cg._nodes.get(func.name)
                    if caller_node:
                        caller_node.call_sites.append(call_site)

                    for target in call_site.all_targets:
                        # Add callee edge
                        if caller_node:
                            caller_node.callees.add(target)
                        # Ensure target node exists
                        if target not in cg._nodes:
                            cg._nodes[target] = CallGraphNode(
                                name=target, is_external=True,
                            )
                        cg._nodes[target].callers.add(func.name)

        return cg

    def _resolve_call(
        self,
        inst: CallInst,
        caller_name: str,
        module: Module,
        points_to: AndersenAnalysis | None,
    ) -> CallSite:
        """Resolve a call instruction to its target(s)."""
        callee = inst.operands[0] if inst.operands else None
        if callee is None:
            return CallSite(instruction=inst, caller=caller_name, is_indirect=True)

        # Check if it's a direct call (callee has a name matching a function)
        if callee.name and module.get_function(callee.name):
            return CallSite(
                instruction=inst,
                caller=caller_name,
                direct_target=callee.name,
                is_indirect=False,
            )

        if callee.name and module.get_external(callee.name):
            return CallSite(
                instruction=inst,
                caller=caller_name,
                direct_target=callee.name,
                is_indirect=False,
            )

        # Indirect call through function pointer
        indirect_targets: set[str] = set()
        if points_to is not None:
            pts = points_to.get_points_to(callee)
            for loc in pts:
                # Try to resolve location to a function name
                func = module.get_function(loc.name)
                if func:
                    indirect_targets.add(func.name)

        return CallSite(
            instruction=inst,
            caller=caller_name,
            indirect_targets=frozenset(indirect_targets),
            is_indirect=True,
        )

    # ── Accessors ────────────────────────────────────────────────────────

    def get_node(self, name: str) -> CallGraphNode | None:
        return self._nodes.get(name)

    @property
    def nodes(self) -> dict[str, CallGraphNode]:
        return dict(self._nodes)

    @property
    def call_sites(self) -> list[CallSite]:
        return list(self._call_sites)

    @property
    def num_functions(self) -> int:
        return len(self._nodes)

    @property
    def num_call_sites(self) -> int:
        return len(self._call_sites)

    def callees_of(self, name: str) -> set[str]:
        node = self._nodes.get(name)
        return node.callees if node else set()

    def callers_of(self, name: str) -> set[str]:
        node = self._nodes.get(name)
        return node.callers if node else set()

    def leaf_functions(self) -> list[str]:
        return [n for n, node in self._nodes.items() if node.is_leaf]

    def root_functions(self) -> list[str]:
        return [n for n, node in self._nodes.items() if node.is_root]

    # ── Reachability ─────────────────────────────────────────────────────

    def reachable_from(self, name: str) -> set[str]:
        """Return all functions reachable from *name* (transitively)."""
        visited: set[str] = set()
        queue: deque[str] = deque([name])
        visited.add(name)
        while queue:
            current = queue.popleft()
            node = self._nodes.get(current)
            if node:
                for callee in node.callees:
                    if callee not in visited:
                        visited.add(callee)
                        queue.append(callee)
        return visited

    def reaches(self, src: str, dst: str) -> bool:
        """Return True if dst is reachable from src."""
        return dst in self.reachable_from(src)

    def reverse_reachable_from(self, name: str) -> set[str]:
        """Return all functions that can reach *name*."""
        visited: set[str] = set()
        queue: deque[str] = deque([name])
        visited.add(name)
        while queue:
            current = queue.popleft()
            node = self._nodes.get(current)
            if node:
                for caller in node.callers:
                    if caller not in visited:
                        visited.add(caller)
                        queue.append(caller)
        return visited

    # ── SCC computation ──────────────────────────────────────────────────

    @property
    def sccs(self) -> list[CallGraphSCC]:
        if self._sccs is None:
            self._compute_sccs()
        return list(self._sccs or [])

    def _compute_sccs(self) -> None:
        """Compute SCCs using Tarjan's algorithm."""
        index_counter = [0]
        stack: list[str] = []
        on_stack: set[str] = set()
        indices: dict[str, int] = {}
        lowlinks: dict[str, int] = {}
        result: list[CallGraphSCC] = []

        def strongconnect(name: str) -> None:
            indices[name] = index_counter[0]
            lowlinks[name] = index_counter[0]
            index_counter[0] += 1
            stack.append(name)
            on_stack.add(name)

            node = self._nodes.get(name)
            if node:
                for callee in node.callees:
                    if callee not in indices:
                        strongconnect(callee)
                        lowlinks[name] = min(lowlinks[name], lowlinks[callee])
                    elif callee in on_stack:
                        lowlinks[name] = min(lowlinks[name], indices[callee])

            if lowlinks[name] == indices[name]:
                component: set[str] = set()
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    component.add(w)
                    if w == name:
                        break

                # Determine if recursive
                is_recursive = len(component) > 1
                if len(component) == 1:
                    func_name = next(iter(component))
                    node = self._nodes.get(func_name)
                    if node and func_name in node.callees:
                        is_recursive = True  # Self-recursive

                scc_nodes = [
                    self._nodes[n] for n in component if n in self._nodes
                ]
                result.append(CallGraphSCC(
                    functions=frozenset(component),
                    is_recursive=is_recursive,
                    nodes=scc_nodes,
                ))

        for name in self._nodes:
            if name not in indices:
                strongconnect(name)

        self._sccs = result

    # ── Topological ordering ─────────────────────────────────────────────

    @property
    def topological_order(self) -> list[str]:
        """Return functions in reverse topological order (callees before callers).

        Useful for bottom-up analysis.
        """
        if self._topo_order is None:
            self._compute_topo_order()
        return list(self._topo_order or [])

    def _compute_topo_order(self) -> None:
        """Compute topological order using DFS post-order."""
        visited: set[str] = set()
        order: list[str] = []

        def dfs(name: str) -> None:
            visited.add(name)
            node = self._nodes.get(name)
            if node:
                for callee in sorted(node.callees):
                    if callee not in visited:
                        dfs(callee)
            order.append(name)

        for name in sorted(self._nodes.keys()):
            if name not in visited:
                dfs(name)

        self._topo_order = order

    @property
    def bottom_up_order(self) -> list[str]:
        """Callees before callers."""
        return self.topological_order

    @property
    def top_down_order(self) -> list[str]:
        """Callers before callees."""
        return list(reversed(self.topological_order))

    # ── Summary ──────────────────────────────────────────────────────────

    def summary(self) -> str:
        sccs = self.sccs
        recursive_sccs = [s for s in sccs if s.is_recursive]
        lines = [
            f"Call Graph:",
            f"  Functions:       {self.num_functions}",
            f"  Call sites:      {self.num_call_sites}",
            f"  SCCs:            {len(sccs)}",
            f"  Recursive SCCs:  {len(recursive_sccs)}",
            f"  Leaf functions:  {len(self.leaf_functions())}",
            f"  Root functions:  {len(self.root_functions())}",
        ]
        if recursive_sccs:
            lines.append("  Recursive components:")
            for scc in recursive_sccs:
                lines.append(f"    {scc}")
        return "\n".join(lines)

    def to_dot(self) -> str:
        """Generate DOT format for visualization."""
        lines = ["digraph CallGraph {"]
        lines.append("  rankdir=TB;")
        lines.append('  node [shape=box, style=filled, fillcolor=lightyellow];')

        for name, node in sorted(self._nodes.items()):
            attrs = []
            if node.is_external:
                attrs.append('fillcolor=lightgray')
                attrs.append('style="filled,dashed"')
            if node.is_leaf:
                attrs.append('fillcolor=lightgreen')
            attr_str = f" [{', '.join(attrs)}]" if attrs else ""
            lines.append(f'  "{name}"{attr_str};')

        for cs in self._call_sites:
            for target in cs.all_targets:
                style = 'style=dashed' if cs.is_indirect else ''
                if style:
                    lines.append(f'  "{cs.caller}" -> "{target}" [{style}];')
                else:
                    lines.append(f'  "{cs.caller}" -> "{target}";')

        lines.append("}")
        return "\n".join(lines)


# ── Convenience ──────────────────────────────────────────────────────────

def build_call_graph(module: Module) -> CallGraph:
    """Build a call graph from a module."""
    return CallGraph.build(module)


def build_call_graph_with_pta(
    module: Module,
    field_sensitive: bool = True,
) -> CallGraph:
    """Build a call graph with points-to analysis for indirect calls."""
    pta = AndersenAnalysis(field_sensitive=field_sensitive)
    pta.analyze_module(module)
    return CallGraph.build(module, points_to=pta)
