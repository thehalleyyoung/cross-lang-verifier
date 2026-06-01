"""Concrete divergence-class oracle plugins.

Importing this package registers all built-in oracles into the plugin registry.
"""

from . import signed_overflow  # noqa: F401  (registers SignedOverflowOracle)
from . import integer_ub  # noqa: F401  (registers shift / div-by-zero / INT_MIN-div-neg1)

__all__ = ["signed_overflow", "integer_ub"]
