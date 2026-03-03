"""
Analysis package for the Cross-Language Equivalence Verifier.

Provides control flow graph construction and analysis, dataflow analysis
frameworks, alias analysis, and call graph construction.
"""

from .cfg import (
    CFG,
    DominatorTree,
    LoopInfo,
    NaturalLoop,
    CFGEdge,
    EdgeKind,
)
from .dataflow import (
    DataflowDirection,
    DataflowResult,
    ReachingDefinitions,
    LiveVariables,
    AvailableExpressions,
    VeryBusyExpressions,
    ConstantPropagation,
    DefUseChains,
    UseDefChains,
)
from .alias import (
    AliasResult,
    PointsToSet,
    AndersenAnalysis,
    SteensgaardAnalysis,
    AliasQuery,
)
from .callgraph import (
    CallGraph,
    CallSite,
    CallGraphNode,
    CallGraphSCC,
)
from .interprocedural import (
    InterproceduralVerifier,
    InterproceduralResult,
    InterproceduralStatus,
    FunctionPairResult,
    InliningTransform,
)

__all__ = [
    "CFG", "DominatorTree", "LoopInfo", "NaturalLoop", "CFGEdge", "EdgeKind",
    "DataflowDirection", "DataflowResult",
    "ReachingDefinitions", "LiveVariables", "AvailableExpressions",
    "VeryBusyExpressions", "ConstantPropagation", "DefUseChains", "UseDefChains",
    "AliasResult", "PointsToSet", "AndersenAnalysis", "SteensgaardAnalysis", "AliasQuery",
    "CallGraph", "CallSite", "CallGraphNode", "CallGraphSCC",
    "InterproceduralVerifier", "InterproceduralResult", "InterproceduralStatus",
    "FunctionPairResult", "InliningTransform",
]
