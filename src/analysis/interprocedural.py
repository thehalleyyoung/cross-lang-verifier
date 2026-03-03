"""
Interprocedural analysis for cross-language equivalence verification.

Provides call-chain-aware verification that propagates equivalence results
across function boundaries. When a callee pair has been verified equivalent,
its calls in the caller can be replaced by their specification (return
value equality), strengthening the caller-level verification.

This implements the compositionality theorem: if g_C ≡ g_R, then for
verifying f_C ≡ f_R we can substitute g_C/g_R calls with a shared
specification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Any

from ..ir.module import Module
from ..ir.function import Function
from ..ir.instructions import CallInst, Instruction
from .callgraph import CallGraph, CallGraphNode, CallGraphSCC

_log = logging.getLogger(__name__)


class InterproceduralStatus(Enum):
    """Status of a function pair in interprocedural verification."""
    UNKNOWN = auto()
    EQUIVALENT = auto()
    DIVERGENT = auto()
    PENDING = auto()
    IN_PROGRESS = auto()
    SKIPPED = auto()


@dataclass
class FunctionPairResult:
    """Result of verifying a single function pair."""
    c_name: str
    rust_name: str
    status: InterproceduralStatus = InterproceduralStatus.UNKNOWN
    confidence: float = 0.0
    callee_deps: List[str] = field(default_factory=list)
    callee_results: Dict[str, InterproceduralStatus] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


@dataclass
class InterproceduralResult:
    """Complete interprocedural verification result."""
    pair_results: Dict[str, FunctionPairResult] = field(default_factory=dict)
    verification_order: List[str] = field(default_factory=list)
    total_pairs: int = 0
    equivalent_pairs: int = 0
    divergent_pairs: int = 0
    unknown_pairs: int = 0

    @property
    def accuracy(self) -> float:
        decided = self.equivalent_pairs + self.divergent_pairs
        return decided / max(self.total_pairs, 1)

    def summary(self) -> str:
        lines = [
            f"Interprocedural Verification: {self.total_pairs} pairs",
            f"  Equivalent: {self.equivalent_pairs}",
            f"  Divergent:  {self.divergent_pairs}",
            f"  Unknown:    {self.unknown_pairs}",
            f"  Accuracy:   {self.accuracy:.1%}",
            f"  Order: {' → '.join(self.verification_order[:10])}",
        ]
        return "\n".join(lines)


class InterproceduralVerifier:
    """
    Interprocedural equivalence verifier.

    Verifies function pairs in bottom-up order: leaf functions first,
    then callers. When a callee pair is equivalent, its calls in the
    caller are strengthened (return value equality assumed).

    Algorithm:
    1. Build call graphs for both C and Rust modules
    2. Match functions between modules (by name)
    3. Compute verification order (topological, leaves first)
    4. Verify each pair, propagating callee equivalence
    """

    def __init__(
        self,
        c_module: Module,
        rust_module: Module,
        verify_fn=None,
        max_inline_depth: int = 3,
    ):
        self._c_module = c_module
        self._rust_module = rust_module
        self._verify_fn = verify_fn  # Callable(c_func, r_func, callee_specs) -> (status, confidence)
        self._max_inline_depth = max_inline_depth
        self._results: Dict[str, FunctionPairResult] = {}

    def verify_all(self) -> InterproceduralResult:
        """Verify all matched function pairs in bottom-up order."""
        c_cg = CallGraph.build(self._c_module)
        r_cg = CallGraph.build(self._rust_module)

        # Match functions by name
        c_funcs = {f.name: f for f in self._c_module.iter_functions()}
        r_funcs = {f.name: f for f in self._rust_module.iter_functions()}
        matched = set(c_funcs.keys()) & set(r_funcs.keys())

        if not matched:
            # Try fuzzy matching: strip underscores, lowercase
            c_norm = {_normalize(n): n for n in c_funcs}
            r_norm = {_normalize(n): n for n in r_funcs}
            for norm_name in set(c_norm.keys()) & set(r_norm.keys()):
                c_name = c_norm[norm_name]
                r_name = r_norm[norm_name]
                matched.add(c_name)  # Use C name as canonical

        # Compute verification order (bottom-up from leaves)
        order = self._compute_order(c_cg, matched)

        result = InterproceduralResult(
            total_pairs=len(matched),
            verification_order=order,
        )

        # Verify each pair in order
        callee_specs: Dict[str, InterproceduralStatus] = {}
        for func_name in order:
            if func_name not in c_funcs or func_name not in r_funcs:
                continue

            c_func = c_funcs[func_name]
            r_func = r_funcs[func_name]

            # Collect callee dependency info
            c_callees = self._get_callees(c_func)
            pair_result = FunctionPairResult(
                c_name=func_name,
                rust_name=func_name,
                callee_deps=list(c_callees & matched),
            )

            # Check callee equivalence
            all_callees_equiv = True
            for callee in pair_result.callee_deps:
                callee_status = callee_specs.get(callee, InterproceduralStatus.UNKNOWN)
                pair_result.callee_results[callee] = callee_status
                if callee_status != InterproceduralStatus.EQUIVALENT:
                    all_callees_equiv = False

            # Verify this pair
            if self._verify_fn:
                try:
                    status, confidence = self._verify_fn(
                        c_func, r_func, callee_specs
                    )
                    pair_result.status = status
                    pair_result.confidence = confidence
                except Exception as e:
                    pair_result.status = InterproceduralStatus.UNKNOWN
                    pair_result.confidence = 0.0
                    pair_result.notes.append(f"Verification error: {e}")
            else:
                # Without a verify function, use structural comparison
                pair_result.status = self._structural_compare(c_func, r_func)
                pair_result.confidence = 0.6 if pair_result.status == InterproceduralStatus.EQUIVALENT else 0.0

            # Boost confidence if all callees are equivalent
            if all_callees_equiv and pair_result.callee_deps:
                pair_result.confidence = min(pair_result.confidence * 1.2, 1.0)
                pair_result.notes.append("All callee dependencies verified equivalent")

            callee_specs[func_name] = pair_result.status
            self._results[func_name] = pair_result
            result.pair_results[func_name] = pair_result

            if pair_result.status == InterproceduralStatus.EQUIVALENT:
                result.equivalent_pairs += 1
            elif pair_result.status == InterproceduralStatus.DIVERGENT:
                result.divergent_pairs += 1
            else:
                result.unknown_pairs += 1

        return result

    def _compute_order(self, cg: CallGraph, matched: Set[str]) -> List[str]:
        """Compute bottom-up verification order using topological sort."""
        # Use the callgraph's topological order if available
        try:
            topo = cg.topological_order()
            # Reverse: verify leaves (callees) before callers
            order = [f for f in reversed(topo) if f in matched]
            # Add any matched functions not in the callgraph
            remaining = matched - set(order)
            order.extend(sorted(remaining))
            return order
        except Exception:
            # Fallback: sort by callee count (leaf functions first)
            c_funcs = {f.name: f for f in self._c_module.iter_functions()}
            def callee_count(name):
                f = c_funcs.get(name)
                if f is None:
                    return 0
                return sum(1 for inst in f.iter_instructions()
                           if isinstance(inst, CallInst))
            return sorted(matched, key=callee_count)

    def _get_callees(self, func: Function) -> Set[str]:
        """Get direct callees of a function."""
        callees = set()
        for inst in func.iter_instructions():
            if isinstance(inst, CallInst):
                name = getattr(inst, 'callee_name', None) or getattr(inst, 'target_name', None)
                if name:
                    callees.add(name)
        return callees

    def _structural_compare(self, c_func: Function, r_func: Function) -> InterproceduralStatus:
        """Quick structural comparison as fallback."""
        c_insts = list(c_func.iter_instructions())
        r_insts = list(r_func.iter_instructions())

        if not c_insts or not r_insts:
            return InterproceduralStatus.UNKNOWN

        # Compare instruction counts
        if abs(len(c_insts) - len(r_insts)) / max(len(c_insts), len(r_insts)) > 0.5:
            return InterproceduralStatus.UNKNOWN

        # Compare opcode histograms
        c_ops = {}
        for inst in c_insts:
            op = type(inst).__name__
            c_ops[op] = c_ops.get(op, 0) + 1

        r_ops = {}
        for inst in r_insts:
            op = type(inst).__name__
            r_ops[op] = r_ops.get(op, 0) + 1

        all_ops = set(c_ops.keys()) | set(r_ops.keys())
        if not all_ops:
            return InterproceduralStatus.UNKNOWN

        intersection = sum(min(c_ops.get(k, 0), r_ops.get(k, 0)) for k in all_ops)
        union = sum(max(c_ops.get(k, 0), r_ops.get(k, 0)) for k in all_ops)
        jaccard = intersection / max(union, 1)

        if jaccard > 0.8:
            return InterproceduralStatus.EQUIVALENT
        return InterproceduralStatus.UNKNOWN


def _normalize(name: str) -> str:
    """Normalize function name for fuzzy matching."""
    # Strip leading underscores
    n = name.lstrip('_')
    # Lowercase
    n = n.lower()
    # Strip common prefixes/suffixes
    for prefix in ('c_', 'rust_', 'r_'):
        if n.startswith(prefix):
            n = n[len(prefix):]
    return n


class InliningTransform:
    """
    Inline simple call chains for interprocedural verification.

    For simple non-recursive callees (leaf functions, small body),
    inline the callee's body into the caller's IR to enable
    end-to-end verification of the call chain.
    """

    MAX_INLINE_SIZE = 50  # Max instructions to inline
    MAX_INLINE_DEPTH = 3  # Max nesting depth

    def __init__(self, module: Module):
        self._module = module
        self._cg = CallGraph.build(module)
        self._inlined: Set[str] = set()

    def should_inline(self, callee_name: str, depth: int = 0) -> bool:
        """Decide whether a callee should be inlined."""
        if depth >= self.MAX_INLINE_DEPTH:
            return False
        node = self._cg.get_node(callee_name) if hasattr(self._cg, 'get_node') else None
        if node is None:
            return False
        if node.is_external:
            return False
        if node.function is None:
            return False
        # Don't inline recursive functions
        if callee_name in node.callees:
            return False
        # Size check
        inst_count = sum(1 for _ in node.function.iter_instructions())
        return inst_count <= self.MAX_INLINE_SIZE

    def get_inlinable_callees(self, func_name: str) -> List[str]:
        """Get list of callees that can be inlined."""
        node = self._cg.get_node(func_name) if hasattr(self._cg, 'get_node') else None
        if node is None:
            return []
        return [c for c in node.callees if self.should_inline(c)]

    def summary(self) -> str:
        """Summary of inlining decisions."""
        lines = ["Inlining Transform Summary"]
        for func in self._module.iter_functions():
            inlinable = self.get_inlinable_callees(func.name)
            if inlinable:
                lines.append(f"  {func.name}: inline [{', '.join(inlinable)}]")
        return "\n".join(lines)
