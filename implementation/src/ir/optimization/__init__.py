"""
IR Optimization Passes for the Cross-Language Equivalence Verifier.

Provides a comprehensive optimization pipeline including:
- Pass management infrastructure (PassManager, AnalysisManager)
- Dead code elimination (DCE, aggressive DCE)
- Constant folding and propagation
- Memory-to-register promotion (mem2reg / SSA construction)
- Instruction simplification and strength reduction
- Function inlining with cost model
- Global value numbering (GVN)
- Loop-invariant code motion (LICM)
- Sparse conditional constant propagation (SCCP)
"""

from .pass_manager import (
    Pass,
    FunctionPass,
    ModulePass,
    PassManager,
    AnalysisManager,
    PassPipeline,
    PassStatistics,
    PassResult,
    create_pipeline_O0,
    create_pipeline_O1,
    create_pipeline_O2,
)
from .dce import (
    DeadCodeElimination,
    AggressiveDCE,
    DeadBlockElimination,
    UnreachableCodeRemoval,
)
from .constant_fold import (
    ConstantFolder,
    ConstantPropagation,
    ConditionalConstantPropagation,
)
from .mem2reg import (
    Mem2Reg,
    PromotableAllocaAnalysis,
    PhiInsertion,
    SSARenamer,
)
from .simplify import (
    InstructionSimplifier,
    AlgebraicSimplification,
    StrengthReduction,
    BooleanSimplification,
    RedundantCastElimination,
)
from .inline import (
    FunctionInliner,
    InlineCostModel,
    InlineDecision,
)
from .gvn import (
    GlobalValueNumbering,
    ValueTable,
    CongruenceClass,
)
from .licm import (
    LoopInvariantCodeMotion,
    LoopInvariantAnalysis,
)
from .sccp import (
    SparseConditionalConstantPropagation,
    LatticeValue,
    SCCPSolver,
)

__all__ = [
    "Pass", "FunctionPass", "ModulePass",
    "PassManager", "AnalysisManager", "PassPipeline", "PassStatistics", "PassResult",
    "create_pipeline_O0", "create_pipeline_O1", "create_pipeline_O2",
    "DeadCodeElimination", "AggressiveDCE", "DeadBlockElimination", "UnreachableCodeRemoval",
    "ConstantFolder", "ConstantPropagation", "ConditionalConstantPropagation",
    "Mem2Reg", "PromotableAllocaAnalysis", "PhiInsertion", "SSARenamer",
    "InstructionSimplifier", "AlgebraicSimplification", "StrengthReduction",
    "BooleanSimplification", "RedundantCastElimination",
    "FunctionInliner", "InlineCostModel", "InlineDecision",
    "GlobalValueNumbering", "ValueTable", "CongruenceClass",
    "LoopInvariantCodeMotion", "LoopInvariantAnalysis",
    "SparseConditionalConstantPropagation", "LatticeValue", "SCCPSolver",
]
