"""Concrete divergence-class oracle plugins.

Importing this package registers all built-in oracles into the plugin registry.
"""

from . import signed_overflow  # noqa: F401  (registers SignedOverflowOracle)
from . import integer_ub  # noqa: F401  (registers shift / div-by-zero / INT_MIN-div-neg1)
from . import memory_shape  # noqa: F401  (registers array-OOB / strict-aliasing)
from . import floating_point  # noqa: F401  (registers FP-contraction)
from . import target_pairs  # noqa: F401  (registers C->Go and C->Swift pairs)
from . import c_to_go  # noqa: F401  (back-compat shim exposing the Go oracles)

__all__ = ["signed_overflow", "integer_ub", "memory_shape", "floating_point",
           "target_pairs", "c_to_go"]
