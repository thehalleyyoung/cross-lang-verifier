"""Utility modules for the Cross-Language Equivalence Verifier."""

from .source_location import SourceLocation, SourceRange, SourceFile
from .diagnostics import Diagnostic, DiagnosticLevel, DiagnosticCollection
from .graph_utils import topological_sort, tarjan_scc, bfs, dfs, to_dot
from .bit_utils import (
    sign_extend, zero_extend, detect_overflow_add, detect_overflow_sub,
    detect_overflow_mul, twos_complement, twos_complement_inverse,
)
from .config_utils import deep_merge, expand_env_vars, resolve_path

__all__ = [
    "SourceLocation", "SourceRange", "SourceFile",
    "Diagnostic", "DiagnosticLevel", "DiagnosticCollection",
    "topological_sort", "tarjan_scc", "bfs", "dfs", "to_dot",
    "sign_extend", "zero_extend", "detect_overflow_add",
    "detect_overflow_sub", "detect_overflow_mul",
    "twos_complement", "twos_complement_inverse",
    "deep_merge", "expand_env_vars", "resolve_path",
]
