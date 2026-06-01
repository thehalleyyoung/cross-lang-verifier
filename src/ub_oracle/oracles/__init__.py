"""Concrete divergence-class oracle plugins.

Importing this package registers all built-in oracles into the plugin registry.
"""

from . import signed_overflow  # noqa: F401  (registers SignedOverflowOracle)

__all__ = ["signed_overflow"]
