"""
Coverage tracking for differential fuzzing.

Tracks basic block coverage, edge coverage, and path coverage.
Manages coverage bitmaps, detects coverage novelty, and generates
coverage reports.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Tuple, Set, Any, FrozenSet


# ---------------------------------------------------------------------------
# Coverage types
# ---------------------------------------------------------------------------

class CoverageKind(Enum):
    """Type of coverage being tracked."""
    BLOCK = auto()
    EDGE = auto()
    PATH = auto()


@dataclass
class CoverageEdge:
    """A control flow edge."""
    src_block: str
    dst_block: str

    @property
    def key(self) -> Tuple[str, str]:
        return (self.src_block, self.dst_block)

    def __hash__(self) -> int:
        return hash(self.key)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CoverageEdge):
            return False
        return self.key == other.key

    def __repr__(self) -> str:
        return f"{self.src_block}→{self.dst_block}"


@dataclass
class CoveragePath:
    """A complete execution path."""
    blocks: Tuple[str, ...]
    edges: Tuple[CoverageEdge, ...]

    @property
    def length(self) -> int:
        return len(self.blocks)

    @property
    def hash(self) -> str:
        return hashlib.sha256(
            ":".join(self.blocks).encode()
        ).hexdigest()[:16]

    def __hash__(self) -> int:
        return hash(self.blocks)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CoveragePath):
            return False
        return self.blocks == other.blocks


# ---------------------------------------------------------------------------
# Coverage map
# ---------------------------------------------------------------------------

class CoverageMap:
    """
    Coverage bitmap for tracking which blocks/edges have been covered.
    
    Uses a fixed-size bitmap hashed from block/edge identifiers,
    similar to AFL's coverage tracking.
    """

    def __init__(self, size: int = 65536):
        self.size = size
        self._bitmap: bytearray = bytearray(size)
        self._total_bits: int = 0
        self._edge_set: Set[Tuple[str, str]] = set()
        self._block_set: Set[str] = set()
        self._path_hashes: Set[str] = set()

    def _hash_key(self, key: str) -> int:
        """Hash a key to a bitmap index."""
        h = hashlib.md5(key.encode()).digest()
        return int.from_bytes(h[:4], 'little') % self.size

    def record_block(self, block_name: str) -> bool:
        """Record a block visit. Returns True if this is new coverage."""
        is_new = block_name not in self._block_set
        self._block_set.add(block_name)

        idx = self._hash_key(f"block:{block_name}")
        if self._bitmap[idx] == 0:
            self._bitmap[idx] = 1
            self._total_bits += 1
            return True
        # Track hit count buckets (1, 2, 4, 8, 16, 32, 64, 128+)
        count = self._bitmap[idx]
        if count < 255:
            self._bitmap[idx] = count + 1
            bucket = self._hit_bucket(count + 1)
            old_bucket = self._hit_bucket(count)
            if bucket != old_bucket:
                return True
        return is_new

    def record_edge(self, src: str, dst: str) -> bool:
        """Record an edge. Returns True if this is new coverage."""
        edge = (src, dst)
        is_new = edge not in self._edge_set
        self._edge_set.add(edge)

        idx = self._hash_key(f"edge:{src}→{dst}")
        if self._bitmap[idx] == 0:
            self._bitmap[idx] = 1
            self._total_bits += 1
            return True
        count = self._bitmap[idx]
        if count < 255:
            self._bitmap[idx] = count + 1
        return is_new

    def record_path(self, blocks: List[str]) -> bool:
        """Record a complete path. Returns True if new."""
        path_hash = hashlib.sha256(":".join(blocks).encode()).hexdigest()[:16]
        is_new = path_hash not in self._path_hashes
        self._path_hashes.add(path_hash)
        return is_new

    def _hit_bucket(self, count: int) -> int:
        """Map hit count to a bucket (AFL-style)."""
        if count == 0:
            return 0
        if count == 1:
            return 1
        if count == 2:
            return 2
        if count == 3:
            return 3
        if count <= 7:
            return 4
        if count <= 15:
            return 5
        if count <= 31:
            return 6
        if count <= 127:
            return 7
        return 8

    @property
    def covered_blocks(self) -> Set[str]:
        return set(self._block_set)

    @property
    def covered_edges(self) -> Set[Tuple[str, str]]:
        return set(self._edge_set)

    @property
    def num_blocks(self) -> int:
        return len(self._block_set)

    @property
    def num_edges(self) -> int:
        return len(self._edge_set)

    @property
    def num_paths(self) -> int:
        return len(self._path_hashes)

    @property
    def bitmap_density(self) -> float:
        """Fraction of bitmap entries that are non-zero."""
        if self.size == 0:
            return 0.0
        return self._total_bits / self.size

    def has_new_coverage(self, other: CoverageMap) -> bool:
        """Check if other map has coverage not in this map."""
        for i in range(self.size):
            if other._bitmap[i] > 0 and self._bitmap[i] == 0:
                return True
        return False

    def merge(self, other: CoverageMap) -> int:
        """Merge another coverage map into this one. Returns count of new entries."""
        new_count = 0
        for i in range(min(self.size, other.size)):
            if other._bitmap[i] > 0 and self._bitmap[i] == 0:
                new_count += 1
            self._bitmap[i] = max(self._bitmap[i], other._bitmap[i])

        self._block_set.update(other._block_set)
        self._edge_set.update(other._edge_set)
        self._path_hashes.update(other._path_hashes)
        self._total_bits = sum(1 for b in self._bitmap if b > 0)
        return new_count

    def copy(self) -> CoverageMap:
        """Create a copy of this coverage map."""
        new_map = CoverageMap(self.size)
        new_map._bitmap = bytearray(self._bitmap)
        new_map._total_bits = self._total_bits
        new_map._block_set = set(self._block_set)
        new_map._edge_set = set(self._edge_set)
        new_map._path_hashes = set(self._path_hashes)
        return new_map

    def reset(self) -> None:
        """Reset all coverage data."""
        self._bitmap = bytearray(self.size)
        self._total_bits = 0
        self._block_set.clear()
        self._edge_set.clear()
        self._path_hashes.clear()


# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------

@dataclass
class CoverageReport:
    """Coverage analysis report."""
    total_blocks: int = 0
    covered_blocks: int = 0
    total_edges: int = 0
    covered_edges: int = 0
    total_paths: int = 0
    uncovered_blocks: List[str] = field(default_factory=list)
    uncovered_edges: List[Tuple[str, str]] = field(default_factory=list)
    hottest_blocks: List[Tuple[str, int]] = field(default_factory=list)

    @property
    def block_coverage_pct(self) -> float:
        if self.total_blocks == 0:
            return 100.0
        return (self.covered_blocks / self.total_blocks) * 100.0

    @property
    def edge_coverage_pct(self) -> float:
        if self.total_edges == 0:
            return 100.0
        return (self.covered_edges / self.total_edges) * 100.0

    def summary(self) -> str:
        lines = [
            "Coverage Report:",
            f"  Blocks: {self.covered_blocks}/{self.total_blocks} ({self.block_coverage_pct:.1f}%)",
            f"  Edges:  {self.covered_edges}/{self.total_edges} ({self.edge_coverage_pct:.1f}%)",
            f"  Paths:  {self.total_paths}",
        ]
        if self.uncovered_blocks:
            lines.append(f"  Uncovered blocks: {', '.join(self.uncovered_blocks[:10])}")
            if len(self.uncovered_blocks) > 10:
                lines.append(f"    ... and {len(self.uncovered_blocks) - 10} more")
        if self.hottest_blocks:
            lines.append("  Hottest blocks:")
            for name, count in self.hottest_blocks[:5]:
                lines.append(f"    {name}: {count} hits")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Coverage tracker
# ---------------------------------------------------------------------------

class CoverageTracker:
    """
    Tracks coverage across fuzzing campaigns.
    
    Maintains separate coverage maps for C and Rust sides,
    detects novelty, and generates reports.
    """

    def __init__(
        self,
        all_blocks: Optional[Set[str]] = None,
        all_edges: Optional[Set[Tuple[str, str]]] = None,
    ):
        self.all_blocks = all_blocks or set()
        self.all_edges = all_edges or set()

        self.c_coverage = CoverageMap()
        self.rust_coverage = CoverageMap()
        self.combined_coverage = CoverageMap()

        self._input_coverage: Dict[str, CoverageMap] = {}

    def record_c_block(self, block: str) -> bool:
        """Record a block covered by the C side."""
        new_c = self.c_coverage.record_block(block)
        new_combined = self.combined_coverage.record_block(f"c:{block}")
        return new_c or new_combined

    def record_rust_block(self, block: str) -> bool:
        """Record a block covered by the Rust side."""
        new_r = self.rust_coverage.record_block(block)
        new_combined = self.combined_coverage.record_block(f"rust:{block}")
        return new_r or new_combined

    def record_c_edge(self, src: str, dst: str) -> bool:
        new_c = self.c_coverage.record_edge(src, dst)
        new_combined = self.combined_coverage.record_edge(f"c:{src}", f"c:{dst}")
        return new_c or new_combined

    def record_rust_edge(self, src: str, dst: str) -> bool:
        new_r = self.rust_coverage.record_edge(src, dst)
        new_combined = self.combined_coverage.record_edge(f"rust:{src}", f"rust:{dst}")
        return new_r or new_combined

    def record_execution(
        self,
        input_id: str,
        c_blocks: List[str],
        rust_blocks: List[str],
        c_edges: Optional[List[Tuple[str, str]]] = None,
        rust_edges: Optional[List[Tuple[str, str]]] = None,
    ) -> bool:
        """Record coverage from a single execution. Returns True if new coverage."""
        has_new = False

        for b in c_blocks:
            if self.record_c_block(b):
                has_new = True

        for b in rust_blocks:
            if self.record_rust_block(b):
                has_new = True

        if c_edges:
            for src, dst in c_edges:
                if self.record_c_edge(src, dst):
                    has_new = True

        if rust_edges:
            for src, dst in rust_edges:
                if self.record_rust_edge(src, dst):
                    has_new = True

        return has_new

    def is_novel(self, c_blocks: List[str], rust_blocks: List[str]) -> bool:
        """Check if the given blocks represent novel coverage."""
        for b in c_blocks:
            if b not in self.c_coverage.covered_blocks:
                return True
        for b in rust_blocks:
            if b not in self.rust_coverage.covered_blocks:
                return True
        return False

    def prioritize_input(
        self,
        c_blocks: List[str],
        rust_blocks: List[str],
    ) -> float:
        """Score an input by how much new coverage it provides."""
        score = 0.0
        for b in c_blocks:
            if b not in self.c_coverage.covered_blocks:
                score += 1.0
        for b in rust_blocks:
            if b not in self.rust_coverage.covered_blocks:
                score += 1.0
        return score

    def generate_report(self) -> CoverageReport:
        """Generate a comprehensive coverage report."""
        report = CoverageReport()

        report.total_blocks = len(self.all_blocks) if self.all_blocks else (
            self.c_coverage.num_blocks + self.rust_coverage.num_blocks
        )
        report.covered_blocks = len(
            self.c_coverage.covered_blocks | self.rust_coverage.covered_blocks
        )
        report.total_edges = len(self.all_edges) if self.all_edges else (
            self.c_coverage.num_edges + self.rust_coverage.num_edges
        )
        report.covered_edges = self.c_coverage.num_edges + self.rust_coverage.num_edges
        report.total_paths = self.c_coverage.num_paths + self.rust_coverage.num_paths

        # Uncovered blocks
        if self.all_blocks:
            covered = self.c_coverage.covered_blocks | self.rust_coverage.covered_blocks
            report.uncovered_blocks = sorted(self.all_blocks - covered)

        # Uncovered edges
        if self.all_edges:
            covered_edges = self.c_coverage.covered_edges | self.rust_coverage.covered_edges
            report.uncovered_edges = sorted(self.all_edges - covered_edges)

        return report

    def summary(self) -> str:
        return self.generate_report().summary()

    def reset(self) -> None:
        """Reset all coverage data."""
        self.c_coverage.reset()
        self.rust_coverage.reset()
        self.combined_coverage.reset()
        self._input_coverage.clear()
