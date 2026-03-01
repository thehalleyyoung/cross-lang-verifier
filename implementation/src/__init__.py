"""
SemRec: Cross-Language Equivalence Verifier

A production tool for verifying semantic equivalence between C and Rust/Go/Zig
programs during migration projects. Uses shared typed SSA IR, product program
construction, bounded symbolic execution with Z3, and differential fuzzing.

Supports whole-project scanning, c2rust integration, incremental CI
verification, and multi-format reporting (HTML, SARIF, GitHub Actions).
"""

__version__ = "0.2.0"
__author__ = "Cross-Language Equivalence Verifier Team"

from . import ir

__all__ = [
    "ir",
    "api",
    "project_scanner",
    "c2rust_integration",
    "incremental_verify",
    "report_generator",
    "language_support",
    "auto_translator",
    "test_migration",
    "dependency_migrator",
    "safety_analysis",
    "migration_dashboard",
    "__version__",
]
