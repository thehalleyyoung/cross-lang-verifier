"""
Symbolic Execution Engine for Cross-Language Equivalence Verification.

Provides bounded symbolic execution of IR programs using Z3 for
constraint solving, with path exploration strategies and symbolic memory.
"""

from .state import (
    SymbolicState,
    SymbolicValue,
    SymbolicMemory,
    PathConstraint,
)
from .executor import (
    SymbolicExecutor,
    ExecutionResult,
    ExecutionConfig,
)
from .path_manager import (
    PathManager,
    PathStrategy,
    PathInfo,
)
from .memory import (
    FlatMemoryModel,
    MemoryRegion,
    MemoryRegionKind,
    MemoryError,
    MemoryErrorKind,
)

__all__ = [
    "SymbolicState",
    "SymbolicValue",
    "SymbolicMemory",
    "PathConstraint",
    "SymbolicExecutor",
    "ExecutionResult",
    "ExecutionConfig",
    "PathManager",
    "PathStrategy",
    "PathInfo",
    "FlatMemoryModel",
    "MemoryRegion",
    "MemoryRegionKind",
    "MemoryError",
    "MemoryErrorKind",
]
