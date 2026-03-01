"""
IR Transformations for the Cross-Language Equivalence Verifier.

Provides structural transformations on the IR including:
- Loop transformations (unrolling, peeling, rotation, unswitching)
- CFG transformations (block merging, splitting, jump threading)
- SSA transformations (construction, deconstruction, copy propagation)
"""

from .loop_transforms import (
    LoopUnroller,
    LoopPeeler,
    LoopRotator,
    LoopUnswitcher,
    LoopVectorizationPrep,
    UnrollStrategy,
)
from .cfg_transforms import (
    BlockMerger,
    BlockSplitter,
    CriticalEdgeSplitter,
    JumpThreader,
    TailDuplicator,
    UnreachableBlockEliminator,
    EdgeProfiler,
)
from .ssa_transforms import (
    SSAConstructor,
    SSADeconstructor,
    CopyPropagator,
    PhiSimplifier,
    ValueRenamer,
)

__all__ = [
    "LoopUnroller", "LoopPeeler", "LoopRotator", "LoopUnswitcher",
    "LoopVectorizationPrep", "UnrollStrategy",
    "BlockMerger", "BlockSplitter", "CriticalEdgeSplitter",
    "JumpThreader", "TailDuplicator", "UnreachableBlockEliminator", "EdgeProfiler",
    "SSAConstructor", "SSADeconstructor", "CopyPropagator",
    "PhiSimplifier", "ValueRenamer",
]
