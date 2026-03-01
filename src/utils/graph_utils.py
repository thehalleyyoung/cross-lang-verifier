"""Graph utilities: topological sort, SCC, BFS/DFS, DOT serialization.

Generic graph algorithms that work on adjacency-list representations.
"""

from __future__ import annotations

from collections import deque, defaultdict
from typing import (
    TypeVar, Generic, Dict, List, Set, Sequence, Callable, Optional,
    Tuple, Any, Iterator,
)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Adjacency representations
# ---------------------------------------------------------------------------

class AdjacencyList(Generic[T]):
    """Directed graph stored as adjacency lists."""

    def __init__(self):
        self._adj: Dict[T, List[T]] = defaultdict(list)
        self._nodes: Set[T] = set()

    def add_node(self, node: T) -> None:
        self._nodes.add(node)
        if node not in self._adj:
            self._adj[node] = []

    def add_edge(self, src: T, dst: T) -> None:
        self._nodes.add(src)
        self._nodes.add(dst)
        self._adj[src].append(dst)

    def remove_edge(self, src: T, dst: T) -> None:
        if src in self._adj:
            try:
                self._adj[src].remove(dst)
            except ValueError:
                pass

    @property
    def nodes(self) -> Set[T]:
        return set(self._nodes)

    def successors(self, node: T) -> List[T]:
        return list(self._adj.get(node, []))

    def predecessors(self, node: T) -> List[T]:
        return [n for n in self._nodes if node in self._adj.get(n, [])]

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    @property
    def num_edges(self) -> int:
        return sum(len(s) for s in self._adj.values())

    def has_edge(self, src: T, dst: T) -> bool:
        return dst in self._adj.get(src, [])

    def in_degree(self, node: T) -> int:
        return sum(1 for n in self._nodes if node in self._adj.get(n, []))

    def out_degree(self, node: T) -> int:
        return len(self._adj.get(node, []))

    def reverse(self) -> AdjacencyList[T]:
        rev = AdjacencyList[T]()
        for n in self._nodes:
            rev.add_node(n)
        for src, dsts in self._adj.items():
            for dst in dsts:
                rev.add_edge(dst, src)
        return rev

    def subgraph(self, nodes: Set[T]) -> AdjacencyList[T]:
        g = AdjacencyList[T]()
        for n in nodes:
            if n in self._nodes:
                g.add_node(n)
        for src in nodes:
            for dst in self._adj.get(src, []):
                if dst in nodes:
                    g.add_edge(src, dst)
        return g

    def to_matrix(self) -> Tuple[List[T], List[List[bool]]]:
        """Convert to adjacency matrix. Returns (node_list, matrix)."""
        node_list = sorted(self._nodes, key=str)
        idx = {n: i for i, n in enumerate(node_list)}
        size = len(node_list)
        matrix = [[False] * size for _ in range(size)]
        for src, dsts in self._adj.items():
            for dst in dsts:
                matrix[idx[src]][idx[dst]] = True
        return node_list, matrix

    @staticmethod
    def from_matrix(nodes: List[T], matrix: List[List[bool]]) -> AdjacencyList[T]:
        g = AdjacencyList[T]()
        for n in nodes:
            g.add_node(n)
        for i, row in enumerate(matrix):
            for j, val in enumerate(row):
                if val:
                    g.add_edge(nodes[i], nodes[j])
        return g

    @staticmethod
    def from_edges(edges: Sequence[Tuple[T, T]]) -> AdjacencyList[T]:
        g = AdjacencyList[T]()
        for src, dst in edges:
            g.add_edge(src, dst)
        return g


# ---------------------------------------------------------------------------
# Topological sort
# ---------------------------------------------------------------------------

class CycleError(Exception):
    """Raised when a cycle is detected during topological sort."""
    def __init__(self, cycle: List = None):
        self.cycle = cycle or []
        super().__init__(f"Graph contains a cycle: {self.cycle}")


def topological_sort(graph: AdjacencyList[T]) -> List[T]:
    """Kahn's algorithm for topological sort. Raises CycleError on cycles."""
    in_deg: Dict[T, int] = {n: 0 for n in graph.nodes}
    for n in graph.nodes:
        for s in graph.successors(n):
            in_deg[s] = in_deg.get(s, 0) + 1

    queue = deque(n for n in graph.nodes if in_deg[n] == 0)
    result: List[T] = []
    while queue:
        node = queue.popleft()
        result.append(node)
        for s in graph.successors(node):
            in_deg[s] -= 1
            if in_deg[s] == 0:
                queue.append(s)

    if len(result) != graph.num_nodes:
        raise CycleError()
    return result


def topological_sort_grouped(graph: AdjacencyList[T]) -> List[List[T]]:
    """Topological sort returning nodes grouped by level (parallelizable)."""
    in_deg: Dict[T, int] = {n: 0 for n in graph.nodes}
    for n in graph.nodes:
        for s in graph.successors(n):
            in_deg[s] = in_deg.get(s, 0) + 1

    current = [n for n in graph.nodes if in_deg[n] == 0]
    levels: List[List[T]] = []
    processed = 0
    while current:
        levels.append(list(current))
        processed += len(current)
        next_level: List[T] = []
        for node in current:
            for s in graph.successors(node):
                in_deg[s] -= 1
                if in_deg[s] == 0:
                    next_level.append(s)
        current = next_level

    if processed != graph.num_nodes:
        raise CycleError()
    return levels


# ---------------------------------------------------------------------------
# Strongly connected components (Tarjan's algorithm)
# ---------------------------------------------------------------------------

def tarjan_scc(graph: AdjacencyList[T]) -> List[List[T]]:
    """Tarjan's algorithm for SCCs. Returns components in reverse topological order."""
    index_counter = [0]
    stack: List[T] = []
    on_stack: Set[T] = set()
    index_map: Dict[T, int] = {}
    lowlink: Dict[T, int] = {}
    result: List[List[T]] = []

    def strongconnect(v: T):
        index_map[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)

        for w in graph.successors(v):
            if w not in index_map:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], index_map[w])

        if lowlink[v] == index_map[v]:
            component: List[T] = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                component.append(w)
                if w == v:
                    break
            result.append(component)

    for node in graph.nodes:
        if node not in index_map:
            strongconnect(node)

    return result


def condensation(graph: AdjacencyList[T]) -> Tuple[AdjacencyList[int], List[List[T]]]:
    """Build condensation DAG from SCCs. Returns (dag, component_list)."""
    sccs = tarjan_scc(graph)
    node_to_scc: Dict[T, int] = {}
    for i, comp in enumerate(sccs):
        for n in comp:
            node_to_scc[n] = i

    dag = AdjacencyList[int]()
    for i in range(len(sccs)):
        dag.add_node(i)

    seen_edges: Set[Tuple[int, int]] = set()
    for n in graph.nodes:
        src_scc = node_to_scc[n]
        for s in graph.successors(n):
            dst_scc = node_to_scc[s]
            if src_scc != dst_scc and (src_scc, dst_scc) not in seen_edges:
                dag.add_edge(src_scc, dst_scc)
                seen_edges.add((src_scc, dst_scc))

    return dag, sccs


# ---------------------------------------------------------------------------
# BFS / DFS
# ---------------------------------------------------------------------------

def bfs(graph: AdjacencyList[T], start: T,
        visitor: Optional[Callable[[T, T], None]] = None) -> List[T]:
    """Breadth-first search. Optional visitor(parent, child) callback. Returns visit order."""
    visited: Set[T] = set()
    queue = deque([start])
    visited.add(start)
    order: List[T] = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for s in graph.successors(node):
            if s not in visited:
                visited.add(s)
                queue.append(s)
                if visitor:
                    visitor(node, s)
    return order


def dfs(graph: AdjacencyList[T], start: T,
        pre_visitor: Optional[Callable[[T], None]] = None,
        post_visitor: Optional[Callable[[T], None]] = None) -> List[T]:
    """Depth-first search (iterative). Returns nodes in pre-order."""
    visited: Set[T] = set()
    order: List[T] = []
    stack: List[Tuple[T, bool]] = [(start, False)]

    while stack:
        node, returning = stack.pop()
        if returning:
            if post_visitor:
                post_visitor(node)
            continue
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        if pre_visitor:
            pre_visitor(node)
        stack.append((node, True))
        for s in reversed(graph.successors(node)):
            if s not in visited:
                stack.append((s, False))
    return order


def dfs_recursive(graph: AdjacencyList[T], start: T,
                  pre_visitor: Optional[Callable[[T], None]] = None,
                  post_visitor: Optional[Callable[[T], None]] = None) -> List[T]:
    """Recursive DFS. Caution: may hit recursion limit on large graphs."""
    visited: Set[T] = set()
    order: List[T] = []

    def visit(node: T):
        if node in visited:
            return
        visited.add(node)
        order.append(node)
        if pre_visitor:
            pre_visitor(node)
        for s in graph.successors(node):
            visit(s)
        if post_visitor:
            post_visitor(node)

    visit(start)
    return order


def find_all_paths(graph: AdjacencyList[T], start: T, end: T,
                   max_paths: int = 100) -> List[List[T]]:
    """Find all simple paths from start to end (bounded by max_paths)."""
    paths: List[List[T]] = []
    stack: List[Tuple[T, List[T], Set[T]]] = [(start, [start], {start})]

    while stack and len(paths) < max_paths:
        node, path, visited = stack.pop()
        if node == end:
            paths.append(path)
            continue
        for s in graph.successors(node):
            if s not in visited:
                stack.append((s, path + [s], visited | {s}))
    return paths


def shortest_path(graph: AdjacencyList[T], start: T, end: T) -> Optional[List[T]]:
    """BFS-based shortest path. Returns None if no path."""
    if start == end:
        return [start]
    visited: Set[T] = {start}
    queue: deque[Tuple[T, List[T]]] = deque([(start, [start])])
    while queue:
        node, path = queue.popleft()
        for s in graph.successors(node):
            if s == end:
                return path + [s]
            if s not in visited:
                visited.add(s)
                queue.append((s, path + [s]))
    return None


def reachable(graph: AdjacencyList[T], start: T) -> Set[T]:
    """Return all nodes reachable from start."""
    return set(bfs(graph, start))


def is_dag(graph: AdjacencyList[T]) -> bool:
    """Check if graph is a DAG."""
    try:
        topological_sort(graph)
        return True
    except CycleError:
        return False


# ---------------------------------------------------------------------------
# DOT serialization
# ---------------------------------------------------------------------------

def to_dot(graph: AdjacencyList[T], name: str = "G",
           node_label: Optional[Callable[[T], str]] = None,
           edge_label: Optional[Callable[[T, T], str]] = None,
           node_attrs: Optional[Callable[[T], Dict[str, str]]] = None,
           directed: bool = True) -> str:
    """Serialize graph to Graphviz DOT format."""
    graph_type = "digraph" if directed else "graph"
    edge_op = "->" if directed else "--"
    lines: List[str] = [f"{graph_type} {name} {{"]
    lines.append("  rankdir=TB;")
    lines.append("  node [shape=box, fontname=\"Courier\"];")

    node_ids: Dict[T, str] = {}
    for i, n in enumerate(sorted(graph.nodes, key=str)):
        nid = f"n{i}"
        node_ids[n] = nid
        label = node_label(n) if node_label else str(n)
        label = label.replace('"', '\\"')
        attrs_str = ""
        if node_attrs:
            attrs = node_attrs(n)
            if attrs:
                pairs = ", ".join(f'{k}="{v}"' for k, v in attrs.items())
                attrs_str = f", {pairs}"
        lines.append(f'  {nid} [label="{label}"{attrs_str}];')

    for src in sorted(graph.nodes, key=str):
        for dst in graph.successors(src):
            edge_attrs = ""
            if edge_label:
                lbl = edge_label(src, dst)
                if lbl:
                    edge_attrs = f' [label="{lbl}"]'
            lines.append(f"  {node_ids[src]} {edge_op} {node_ids[dst]}{edge_attrs};")

    lines.append("}")
    return "\n".join(lines)


def from_dot_edges(dot_text: str) -> AdjacencyList[str]:
    """Parse a simple DOT graph extracting edge relationships."""
    graph = AdjacencyList[str]()
    for line in dot_text.splitlines():
        line = line.strip().rstrip(";")
        if "->" in line:
            parts = line.split("->")
            if len(parts) == 2:
                src = parts[0].strip().split("[")[0].strip()
                dst = parts[1].strip().split("[")[0].strip()
                graph.add_edge(src, dst)
        elif "--" in line:
            parts = line.split("--")
            if len(parts) == 2:
                src = parts[0].strip().split("[")[0].strip()
                dst = parts[1].strip().split("[")[0].strip()
                graph.add_edge(src, dst)
                graph.add_edge(dst, src)
    return graph


# ---------------------------------------------------------------------------
# Graph metrics
# ---------------------------------------------------------------------------

def compute_density(graph: AdjacencyList[T]) -> float:
    """Compute edge density = edges / (nodes * (nodes - 1))."""
    n = graph.num_nodes
    if n <= 1:
        return 0.0
    return graph.num_edges / (n * (n - 1))


def compute_diameter(graph: AdjacencyList[T]) -> int:
    """Compute diameter (longest shortest path) via BFS from each node."""
    max_dist = 0
    for node in graph.nodes:
        visited: Set[T] = {node}
        queue = deque([(node, 0)])
        while queue:
            current, dist = queue.popleft()
            max_dist = max(max_dist, dist)
            for s in graph.successors(current):
                if s not in visited:
                    visited.add(s)
                    queue.append((s, dist + 1))
    return max_dist
