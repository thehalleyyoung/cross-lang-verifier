#!/usr/bin/env python3
"""
CEGAR Evaluation Engine for LLM Translation Pipelines.

Implements a counterexample-guided abstraction refinement loop:
1. LLM translates C → Rust
2. Verification oracle checks equivalence
3. If divergent, counterexample + repair hint fed back to LLM
4. Repeat until equivalent or max iterations

This is the core novelty: the first systematic verification oracle
for LLM-based code translation with formal counterexample feedback.
"""

from __future__ import annotations

import json
import time
import os
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CEGARIteration:
    """Record of a single CEGAR iteration."""
    iteration: int
    rust_code: str
    verdict: str  # equivalent, divergent, error, unknown
    counterexample: Optional[Dict[str, Any]] = None
    repair_hint: Optional[str] = None
    divergence_class: str = ""
    time_ms: float = 0.0
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0


@dataclass
class CEGARResult:
    """Complete result of a CEGAR evaluation run."""
    func_name: str
    c_code: str
    converged: bool
    final_verdict: str
    iterations: List[CEGARIteration] = field(default_factory=list)
    total_iterations: int = 0
    total_time_ms: float = 0.0
    bug_class: str = ""  # Classification of the original bug
    llm_repairable: bool = False  # Could the LLM fix it?
    repair_iterations: int = 0  # How many iterations to fix (0 if not fixed)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


# Bug classification taxonomy
BUG_CLASSES = {
    "signed_overflow": {"category": "arithmetic", "typical_fix": "wrapping_op", "difficulty": "easy"},
    "int_min_negation": {"category": "arithmetic", "typical_fix": "wrapping_op", "difficulty": "easy"},
    "int_min_div_neg1": {"category": "division", "typical_fix": "guard", "difficulty": "medium"},
    "division_by_zero": {"category": "division", "typical_fix": "guard", "difficulty": "easy"},
    "shift_ub": {"category": "shift", "typical_fix": "mask", "difficulty": "medium"},
    "shift_negative": {"category": "shift", "typical_fix": "guard", "difficulty": "medium"},
    "cast_truncation": {"category": "cast", "typical_fix": "explicit_cast", "difficulty": "easy"},
    "cast_sign_change": {"category": "cast", "typical_fix": "explicit_cast", "difficulty": "easy"},
    "unsigned_wrap": {"category": "arithmetic", "typical_fix": "wrapping_op", "difficulty": "easy"},
    "output_mismatch": {"category": "semantic", "typical_fix": "logic_fix", "difficulty": "hard"},
    "c_undefined_behavior": {"category": "ub", "typical_fix": "wrapping_op", "difficulty": "medium"},
}

REPAIRABILITY_EASY = {"signed_overflow", "int_min_negation", "division_by_zero",
                       "cast_truncation", "cast_sign_change", "unsigned_wrap"}
REPAIRABILITY_MEDIUM = {"int_min_div_neg1", "shift_ub", "shift_negative",
                         "c_undefined_behavior"}
REPAIRABILITY_HARD = {"output_mismatch"}


def classify_repairability(bug_class: str) -> str:
    """Classify how easily an LLM can repair a given bug class."""
    if bug_class in REPAIRABILITY_EASY:
        return "easy"
    elif bug_class in REPAIRABILITY_MEDIUM:
        return "medium"
    else:
        return "hard"


class CEGAREngine:
    """
    CEGAR loop that uses an LLM to translate C→Rust and a verification
    oracle to check correctness, feeding counterexamples back.
    """

    def __init__(self, model: str = "gpt-4.1-nano", max_iterations: int = 5,
                 timeout_ms: int = 10000, api_key: Optional[str] = None):
        self.model = model
        self.max_iterations = max_iterations
        self.timeout_ms = timeout_ms
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def _call_llm(self, messages: List[Dict[str, str]]) -> Tuple[str, int, int]:
        """Call LLM and return (response_text, prompt_tokens, completion_tokens)."""
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_tokens=1024,
        )
        text = response.choices[0].message.content or ""
        usage = response.usage
        return text, usage.prompt_tokens if usage else 0, usage.completion_tokens if usage else 0

    def _extract_rust_code(self, llm_response: str) -> str:
        """Extract Rust code from LLM response."""
        import re
        m = re.search(r'```rust\s*(.*?)```', llm_response, re.DOTALL)
        if m:
            return m.group(1).strip()
        m = re.search(r'```\s*((?:pub\s+)?fn\s+.*?)```', llm_response, re.DOTALL)
        if m:
            return m.group(1).strip()
        # Try to find bare function (with or without pub)
        m = re.search(r'((?:pub\s+)?fn\s+\w+\s*\(.*?\).*?\{.*\})', llm_response, re.DOTALL)
        if m:
            return m.group(1).strip()
        return llm_response.strip()

    def _clean_rust_code(self, code: str) -> str:
        """Clean LLM-generated Rust code for verification."""
        from src.oracle.oracle import preprocess_rust_code
        return preprocess_rust_code(code)

    def _initial_prompt(self, c_code: str) -> List[Dict[str, str]]:
        """Create initial translation prompt."""
        return [
            {"role": "system", "content": (
                "You are a C-to-Rust translation expert. Translate the given C function "
                "to semantically equivalent Rust. The Rust function must produce identical "
                "output for ALL possible inputs, including edge cases like overflow, "
                "division by zero, and shift by >= bit width. Use wrapping arithmetic "
                "where C has undefined behavior. Return ONLY the Rust function in a "
                "```rust``` code block."
            )},
            {"role": "user", "content": f"Translate this C function to equivalent Rust:\n\n```c\n{c_code}\n```"}
        ]

    def _repair_prompt(self, c_code: str, rust_code: str,
                       counterexample: Dict[str, Any],
                       repair_hint: str, iteration: int) -> List[Dict[str, str]]:
        """Create repair prompt with counterexample feedback."""
        cex_str = json.dumps(counterexample, indent=2)
        return [
            {"role": "system", "content": (
                "You are a C-to-Rust translation expert. A previous translation was "
                "found to be semantically incorrect by a formal verification oracle. "
                "Fix the Rust translation to match C semantics exactly. "
                "Return ONLY the corrected Rust function in a ```rust``` code block."
            )},
            {"role": "user", "content": (
                f"The following C function:\n```c\n{c_code}\n```\n\n"
                f"Was translated to this Rust (iteration {iteration}):\n```rust\n{rust_code}\n```\n\n"
                f"The verification oracle found a DIVERGENCE with this counterexample:\n"
                f"```json\n{cex_str}\n```\n\n"
                f"Repair hint: {repair_hint}\n\n"
                f"Fix the Rust translation to produce identical output to C for ALL inputs."
            )}
        ]

    def _syntax_fix_prompt(self, c_code: str, rust_code: str,
                           error_msg: str) -> List[Dict[str, str]]:
        """Create prompt to fix syntax/parse errors in Rust code."""
        return [
            {"role": "system", "content": (
                "You are a C-to-Rust translation expert. A previous translation "
                "had a syntax error that prevented verification. Rewrite the Rust "
                "function using simple, direct syntax. Avoid closures, method "
                "chains, and complex expressions. Use basic if/else, wrapping "
                "arithmetic (wrapping_add, wrapping_mul, etc.), and simple let "
                "bindings. Return ONLY the Rust function in a ```rust``` code block."
            )},
            {"role": "user", "content": (
                f"The following C function:\n```c\n{c_code}\n```\n\n"
                f"Was translated to this Rust:\n```rust\n{rust_code}\n```\n\n"
                f"But verification failed with: {error_msg}\n\n"
                f"Rewrite the Rust function using simpler syntax that avoids "
                f"the error. Use wrapping arithmetic for C-equivalent semantics."
            )}
        ]

    def run(self, c_code: str, func_name: str = "func") -> CEGARResult:
        """Run the full CEGAR loop on a C function."""
        import sys, os
        impl_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if impl_dir not in sys.path:
            sys.path.insert(0, impl_dir)
        from src.oracle.oracle import VerificationOracle

        oracle = VerificationOracle(timeout_ms=self.timeout_ms)
        start = time.time()
        iterations = []
        converged = False
        final_verdict = "unknown"
        first_bug_class = ""

        # Iteration 0: initial translation
        try:
            messages = self._initial_prompt(c_code)
            llm_response, pt, ct = self._call_llm(messages)
            rust_code = self._extract_rust_code(llm_response)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return CEGARResult(
                func_name=func_name, c_code=c_code, converged=False,
                final_verdict="error", total_time_ms=(time.time() - start) * 1000,
            )

        for i in range(self.max_iterations):
            iter_start = time.time()
            # Clean code before verification
            cleaned_code = self._clean_rust_code(rust_code)
            result = oracle.verify(c_code, cleaned_code, func_name)
            iter_time = (time.time() - iter_start) * 1000

            cex_dict = result.counterexample.to_dict() if result.counterexample else None
            hint_str = result.repair_hint.description if result.repair_hint else ""
            div_class = result.counterexample.divergence_class if result.counterexample else ""

            if i == 0 and div_class:
                first_bug_class = div_class

            # On parse/IR error, treat as needing repair rather than giving up
            if result.verdict == "error" and i < self.max_iterations - 1:
                error_msg = result.error_msg or ""
                if "parse failed" in error_msg or "IR lowering failed" in error_msg:
                    logger.info(f"Verification error on iteration {i}, requesting LLM syntax fix")
                    try:
                        fix_messages = self._syntax_fix_prompt(c_code, rust_code, error_msg)
                        llm_response, pt2, ct2 = self._call_llm(fix_messages)
                        rust_code = self._extract_rust_code(llm_response)
                        it = CEGARIteration(
                            iteration=i, rust_code=rust_code,
                            verdict="error", repair_hint=f"syntax fix: {error_msg}",
                            time_ms=iter_time, llm_prompt_tokens=pt2,
                            llm_completion_tokens=ct2,
                        )
                        iterations.append(it)
                        continue
                    except Exception as e:
                        logger.error(f"LLM syntax fix call failed: {e}")

            it = CEGARIteration(
                iteration=i,
                rust_code=rust_code,
                verdict=result.verdict,
                counterexample=cex_dict,
                repair_hint=hint_str,
                divergence_class=div_class,
                time_ms=iter_time,
                llm_prompt_tokens=pt if i == 0 else 0,
                llm_completion_tokens=ct if i == 0 else 0,
            )
            iterations.append(it)

            if result.verdict == "equivalent":
                converged = True
                final_verdict = "equivalent"
                break
            elif result.verdict in ("error", "unknown"):
                final_verdict = result.verdict
                break
            else:
                # divergent — try repair
                if i < self.max_iterations - 1:
                    try:
                        messages = self._repair_prompt(
                            c_code, rust_code, cex_dict or {}, hint_str, i + 1
                        )
                        llm_response, pt2, ct2 = self._call_llm(messages)
                        rust_code = self._extract_rust_code(llm_response)
                        it.llm_prompt_tokens = pt2
                        it.llm_completion_tokens = ct2
                    except Exception as e:
                        logger.error(f"LLM repair call failed: {e}")
                        final_verdict = "divergent"
                        break
                else:
                    final_verdict = "divergent"

        total_time = (time.time() - start) * 1000
        repair_iters = 0
        if converged:
            repair_iters = len(iterations)

        return CEGARResult(
            func_name=func_name,
            c_code=c_code,
            converged=converged,
            final_verdict=final_verdict,
            iterations=iterations,
            total_iterations=len(iterations),
            total_time_ms=total_time,
            bug_class=first_bug_class or "none",
            llm_repairable=converged,
            repair_iterations=repair_iters,
        )

    def run_batch(self, pairs: List[Tuple[str, str]], 
                  progress_callback=None) -> List[CEGARResult]:
        """Run CEGAR on a batch of (func_name, c_code) pairs."""
        results = []
        for idx, (name, c_code) in enumerate(pairs):
            if progress_callback:
                progress_callback(idx, len(pairs), name)
            try:
                result = self.run(c_code, name)
                results.append(result)
            except Exception as e:
                logger.error(f"CEGAR failed for {name}: {e}")
                results.append(CEGARResult(
                    func_name=name, c_code=c_code, converged=False,
                    final_verdict="error",
                ))
        return results


def analyze_cegar_results(results: List[CEGARResult]) -> Dict[str, Any]:
    """Analyze a batch of CEGAR results for the paper."""
    total = len(results)
    converged = sum(1 for r in results if r.converged)
    divergent = sum(1 for r in results if r.final_verdict == "divergent")
    errors = sum(1 for r in results if r.final_verdict == "error")
    equivalent_initial = sum(1 for r in results 
                            if r.iterations and r.iterations[0].verdict == "equivalent")

    # Bug classification
    bug_counts: Dict[str, int] = {}
    repairability: Dict[str, Dict[str, int]] = {"easy": {}, "medium": {}, "hard": {}}
    for r in results:
        if r.bug_class and r.bug_class != "none":
            bug_counts[r.bug_class] = bug_counts.get(r.bug_class, 0) + 1
            diff = classify_repairability(r.bug_class)
            cat = repairability[diff]
            cat[r.bug_class] = cat.get(r.bug_class, 0) + 1

    # Convergence by iteration
    convergence_curve = {}
    for r in results:
        if r.converged and r.repair_iterations > 0:
            k = r.repair_iterations
            convergence_curve[k] = convergence_curve.get(k, 0) + 1

    # Repairability by class
    repair_by_class: Dict[str, Dict[str, int]] = {}
    for r in results:
        bc = r.bug_class or "none"
        if bc not in repair_by_class:
            repair_by_class[bc] = {"total": 0, "repaired": 0}
        repair_by_class[bc]["total"] += 1
        if r.llm_repairable:
            repair_by_class[bc]["repaired"] += 1

    # Iteration distribution
    iter_counts = [r.total_iterations for r in results]
    avg_iters = sum(iter_counts) / max(len(iter_counts), 1)
    times = [r.total_time_ms for r in results]
    avg_time = sum(times) / max(len(times), 1)

    return {
        "total_pairs": total,
        "converged": converged,
        "convergence_rate": round(converged / max(total, 1) * 100, 1),
        "equivalent_on_first_try": equivalent_initial,
        "remained_divergent": divergent,
        "errors": errors,
        "avg_iterations": round(avg_iters, 2),
        "avg_time_ms": round(avg_time, 1),
        "bug_classification": bug_counts,
        "repairability_breakdown": repairability,
        "convergence_curve": convergence_curve,
        "repair_by_class": repair_by_class,
    }
