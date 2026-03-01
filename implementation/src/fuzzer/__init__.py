"""
Differential Fuzzer for Cross-Language Equivalence Verification.

Provides fuzzing engine, seed generation from divergence tables,
test harness generation, coverage tracking, and input minimization.
"""

from .engine import (
    FuzzEngine,
    FuzzConfig,
    FuzzResult,
    FuzzInput,
    FuzzCampaign,
)
from .seed_generator import (
    SeedGenerator,
    SeedSet,
    BoundaryValueGenerator,
)
from .harness import (
    TestHarness,
    HarnessConfig,
    HarnessGenerator,
)
from .coverage import (
    CoverageTracker,
    CoverageMap,
    CoverageReport,
    CoverageKind,
)
from .minimizer import (
    InputMinimizer,
    MinimizationResult,
    MinimizationStrategy,
)

__all__ = [
    "FuzzEngine",
    "FuzzConfig",
    "FuzzResult",
    "FuzzInput",
    "FuzzCampaign",
    "SeedGenerator",
    "SeedSet",
    "BoundaryValueGenerator",
    "TestHarness",
    "HarnessConfig",
    "HarnessGenerator",
    "CoverageTracker",
    "CoverageMap",
    "CoverageReport",
    "CoverageKind",
    "InputMinimizer",
    "MinimizationResult",
    "MinimizationStrategy",
]
