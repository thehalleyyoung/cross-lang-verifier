#!/usr/bin/env python3
"""
SemRec Verification Oracle — the core API.

Usage:
    from src.oracle.oracle import VerificationOracle
    oracle = VerificationOracle()
    result = oracle.verify("int f(int x){return x+1;}", "fn f(x:i32)->i32{x.wrapping_add(1)}")
    print(result.verdict, result.counterexample, result.repair_hint)

This module is the primary novelty contribution: a composable verification
oracle that any LLM translation pipeline can use as a drop-in correctness
check. It returns structured (verdict, counterexample, repair_hint) triples
that guide automated repair.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Any, Tuple
from enum import Enum, auto


class Verdict(Enum):
    EQUIVALENT = "equivalent"
    DIVERGENT = "divergent"
    UNKNOWN = "unknown"
    ERROR = "error"


@dataclass
class CounterexampleInfo:
    """Structured counterexample from SMT solving."""
    inputs: Dict[str, str] = field(default_factory=dict)
    c_behavior: Optional[str] = None
    rust_behavior: Optional[str] = None
    reason: str = ""
    divergence_class: str = ""  # taxonomy: overflow, division, shift, cast, etc.

    def to_dict(self) -> dict:
        return asdict(self)

    def format_human(self) -> str:
        parts = [f"Divergence: {self.reason}"]
        if self.inputs:
            parts.append(f"  Inputs: {self.inputs}")
        if self.c_behavior:
            parts.append(f"  C behavior: {self.c_behavior}")
        if self.rust_behavior:
            parts.append(f"  Rust behavior: {self.rust_behavior}")
        if self.divergence_class:
            parts.append(f"  Class: {self.divergence_class}")
        return "\n".join(parts)


@dataclass
class RepairHint:
    """Actionable repair guidance derived from the counterexample."""
    description: str = ""
    suggested_fix: str = ""
    fix_category: str = ""  # wrapping_op, checked_op, guard, cast, etc.
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OracleResult:
    """Complete verification result from the oracle."""
    verdict: str  # Verdict enum value
    counterexample: Optional[CounterexampleInfo] = None
    repair_hint: Optional[RepairHint] = None
    time_ms: float = 0.0
    smt_queries: int = 0
    pipeline_stages: Dict[str, bool] = field(default_factory=dict)
    error_msg: Optional[str] = None
    func_name: str = ""
    confidence: float = 1.0

    def to_dict(self) -> dict:
        d = {
            "verdict": self.verdict,
            "time_ms": round(self.time_ms, 2),
            "smt_queries": self.smt_queries,
            "func_name": self.func_name,
            "confidence": self.confidence,
        }
        if self.counterexample:
            d["counterexample"] = self.counterexample.to_dict()
        if self.repair_hint:
            d["repair_hint"] = self.repair_hint.to_dict()
        if self.error_msg:
            d["error_msg"] = self.error_msg
        d["pipeline_stages"] = self.pipeline_stages
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


# ---------------------------------------------------------------------------
# Repair hint generation from counterexamples
# ---------------------------------------------------------------------------

_REPAIR_RULES = [
    {
        "pattern": "signed_overflow",
        "fix_category": "wrapping_op",
        "description": "C signed overflow is UB; the compiler may assume it never happens. Rust panics in debug or wraps in release.",
        "suggested_fix": "Replace ALL signed arithmetic: a + b → a.wrapping_add(b), a - b → a.wrapping_sub(b), a * b → a.wrapping_mul(b). Do this for EVERY +, -, * on i32/i64.",
    },
    {
        "pattern": "int_min_negation",
        "fix_category": "wrapping_op",
        "description": "Negating INT_MIN (-2147483648) is UB in C; -INT_MIN overflows.",
        "suggested_fix": "Replace -x with x.wrapping_neg(). Example: fn negate(x: i32) -> i32 { x.wrapping_neg() }",
    },
    {
        "pattern": "int_min_div_neg1",
        "fix_category": "guard",
        "description": "INT_MIN / -1 is UB in C because the result (2147483648) overflows i32.",
        "suggested_fix": "Add guard before division: if a == i32::MIN && b == -1 { return i32::MIN; } then a / b. Also guard a % b similarly.",
    },
    {
        "pattern": "shift_ub",
        "fix_category": "wrapping_op",
        "description": "Shifting by >= bit width (32 for i32) is UB in C. C compilers typically mask with & 31.",
        "suggested_fix": "Use wrapping_shl/wrapping_shr: x.wrapping_shl(n as u32) which masks automatically. Or manually: x << ((n & 31) as u32).",
    },
    {
        "pattern": "shift_negative",
        "fix_category": "wrapping_op",
        "description": "Left-shifting a negative value is UB in C.",
        "suggested_fix": "Use wrapping_shl: x.wrapping_shl(n as u32). This handles negative values correctly via two's complement.",
    },
    {
        "pattern": "division_by_zero",
        "fix_category": "guard",
        "description": "Division by zero is UB in C; Rust panics.",
        "suggested_fix": "Guard: if b == 0 { return 0; } (or appropriate default) before every a / b and a % b.",
    },
    {
        "pattern": "cast_truncation",
        "fix_category": "cast",
        "description": "Integer narrowing cast may lose bits: (int)long_val truncates in C.",
        "suggested_fix": "Use explicit 'as i32' cast in Rust. Both C and Rust truncate the same way for 'as' casts.",
    },
    {
        "pattern": "cast_sign_change",
        "fix_category": "cast",
        "description": "Signed/unsigned reinterpretation: (unsigned)(-1) == 4294967295 in C.",
        "suggested_fix": "Use 'as u32' for signed→unsigned, 'as i32' for unsigned→signed. Bit pattern is preserved.",
    },
    {
        "pattern": "output_mismatch",
        "fix_category": "wrapping_op",
        "description": "Output differs on specific inputs — likely unsigned overflow semantics.",
        "suggested_fix": "Replace ALL signed arithmetic with wrapping variants: wrapping_add, wrapping_sub, wrapping_mul, wrapping_neg, wrapping_shl, wrapping_shr. Use 'as' casts for type conversions.",
    },
    {
        "pattern": "c_undefined_behavior",
        "fix_category": "wrapping_op",
        "description": "C undefined behavior detected. The C code relies on UB that compilers handle as two's complement.",
        "suggested_fix": "Use wrapping arithmetic for ALL signed operations. C UB on overflow means the Rust code must use wrapping_add/wrapping_sub/wrapping_mul to match two's complement behavior.",
    },
    {
        "pattern": "pointer",
        "fix_category": "bounds_check",
        "description": "Pointer operation divergence: C allows out-of-bounds UB, Rust bounds-checks.",
        "suggested_fix": "Add bounds checking: if idx < arr.len() { arr[idx] } else { default }. Use .get(idx).copied().unwrap_or(default) for safe access.",
    },
    {
        "pattern": "null",
        "fix_category": "option",
        "description": "Null pointer check: C uses NULL, Rust uses Option<&T>.",
        "suggested_fix": "Use Option<&T> and match/unwrap_or for null-pointer patterns.",
    },
]


def generate_repair_hint(reason: str) -> RepairHint:
    """Generate a structured repair hint from a divergence reason."""
    reason_lower = reason.lower()
    for rule in _REPAIR_RULES:
        if rule["pattern"] in reason_lower:
            return RepairHint(
                description=rule["description"],
                suggested_fix=rule["suggested_fix"],
                fix_category=rule["fix_category"],
                confidence=0.8,
            )
    # Generic hint for unrecognized divergences
    return RepairHint(
        description=f"Semantic divergence detected: {reason}",
        suggested_fix="Review the C semantics for undefined behavior and ensure the Rust translation handles all edge cases.",
        fix_category="review",
        confidence=0.3,
    )


def classify_divergence(reason: str) -> str:
    """Classify a divergence reason into the bug taxonomy."""
    r = reason.lower()
    if "overflow" in r or "wrapping" in r:
        return "overflow"
    if "div" in r or "division" in r or "modulo" in r:
        return "division"
    if "shift" in r or "shl" in r or "shr" in r:
        return "shift"
    if "cast" in r or "truncat" in r or "sign_ext" in r:
        return "cast"
    if "negat" in r or "int_min" in r:
        return "overflow"  # INT_MIN negation is overflow class
    if "undefined" in r or "ub" in r:
        return "undefined_behavior"
    return "other"


# ---------------------------------------------------------------------------
# Verification Oracle
# ---------------------------------------------------------------------------

def preprocess_rust_code(rust_code: str) -> str:
    """Preprocess LLM-generated Rust code to handle common patterns.

    LLMs often produce code with minor syntactic variations that our parser
    doesn't handle. This normalizes the code before parsing.
    """
    import re
    code = rust_code.strip()

    # Strip markdown code fences if present
    m = re.match(r'```(?:rust)?\s*\n?(.*?)```', code, re.DOTALL)
    if m:
        code = m.group(1).strip()

    # Strip single-line comments (preserve structure)
    code = re.sub(r'//[^\n]*', '', code)
    # Strip block comments
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)

    # Normalize whitespace: collapse multiple blank lines
    code = re.sub(r'\n{3,}', '\n\n', code)

    # Ensure function has pub or fn at start (LLMs sometimes add extra text)
    lines = code.split('\n')
    fn_start = None
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if re.match(r'(pub\s+)?(unsafe\s+)?(const\s+)?(async\s+)?fn\s+\w+', stripped):
            fn_start = i
            break
    if fn_start is not None and fn_start > 0:
        code = '\n'.join(lines[fn_start:])

    return code.strip()


class VerificationOracle:
    """
    The SemRec Verification Oracle.

    Provides a single-function API for checking C↔Rust semantic equivalence:
        oracle.verify(c_code, rust_code) → OracleResult

    The OracleResult contains:
        - verdict: equivalent / divergent / unknown / error
        - counterexample: concrete inputs witnessing divergence
        - repair_hint: actionable fix guidance for LLM repair loops

    This is designed to be embedded in LLM translation pipelines as a
    correctness oracle, enabling CEGAR-style iterative repair.
    """

    def __init__(self, timeout_ms: int = 10000, func_name: str = "",
                 cegar_mode: bool = False):
        self.timeout_ms = timeout_ms
        self._default_func_name = func_name
        self._cegar_mode = cegar_mode

    def verify(self, c_code: str, rust_code: str,
               func_name: Optional[str] = None) -> OracleResult:
        """Verify semantic equivalence of a C/Rust function pair.

        Args:
            c_code: Complete C function source code.
            rust_code: Complete Rust function source code.
            func_name: Optional function name hint.

        Returns:
            OracleResult with verdict, counterexample, and repair hint.
        """
        import sys, os
        # Ensure implementation is importable
        impl_dir = os.path.join(os.path.dirname(__file__), "..", "..")
        if impl_dir not in sys.path:
            sys.path.insert(0, impl_dir)

        name = func_name or self._default_func_name or self._infer_func_name(c_code)
        start = time.time()

        # Preprocess Rust code to handle common LLM output patterns
        cleaned_rust = preprocess_rust_code(rust_code)

        try:
            return self._run_pipeline(name, c_code, cleaned_rust, start)
        except Exception as e:
            # If preprocessing didn't help, try with original code
            if cleaned_rust != rust_code.strip():
                try:
                    return self._run_pipeline(name, c_code, rust_code.strip(), start)
                except Exception:
                    pass
            return OracleResult(
                verdict=Verdict.ERROR.value,
                error_msg=str(e),
                time_ms=(time.time() - start) * 1000,
                func_name=name,
                confidence=0.0,
            )

    def verify_batch(self, pairs: List[Tuple[str, str, str]]) -> List[OracleResult]:
        """Verify a batch of (name, c_code, rust_code) triples."""
        return [self.verify(c, r, n) for n, c, r in pairs]

    def _infer_func_name(self, c_code: str) -> str:
        """Infer function name from C code."""
        import re
        m = re.search(r'\b(\w+)\s*\(', c_code)
        # Skip type keywords
        skip = {'int', 'unsigned', 'long', 'short', 'char', 'void', 'signed',
                'const', 'static', 'inline', 'extern'}
        if m and m.group(1) not in skip:
            return m.group(1)
        # Try harder: find the last identifier before (
        for m2 in re.finditer(r'(\w+)\s*\(', c_code):
            if m2.group(1) not in skip:
                return m2.group(1)
        return "func"

    def _run_pipeline(self, name: str, c_code: str, rust_code: str,
                      start: float) -> OracleResult:
        """Run the full SemRec pipeline and produce an OracleResult."""
        import z3
        from src.frontend_c.parser import CParser
        from src.frontend_c.ir_lowering import CIRLowering
        from src.frontend_rust.parser import RustParser
        from src.frontend_rust.ir_lowering import RustIRLowering
        from src.frontend_rust.type_resolver import RustTypeResolver
        from src.product_program.product import ProductBuilder
        from src.product_program.alignment import FunctionAligner
        from src.semantics.semantic_config import SemanticConfig
        from src.smt.encoder import SMTEncoder, EncodingContext

        stages = {}

        # Parse C
        try:
            c_ast = CParser(c_code, f"{name}.c").parse()
            stages["c_parse"] = True
        except Exception as e:
            return OracleResult(
                verdict=Verdict.ERROR.value,
                error_msg=f"C parse failed: {e}",
                time_ms=(time.time() - start) * 1000,
                pipeline_stages=stages, func_name=name,
            )

        # Parse Rust — try harder on failure
        try:
            r_ast = RustParser(rust_code, f"{name}.rs").parse()
            stages["rust_parse"] = True
        except Exception as e:
            # Retry: add pub keyword if missing
            import re
            retried = False
            if re.match(r'\s*fn\s+', rust_code):
                try:
                    r_ast = RustParser("pub " + rust_code, f"{name}.rs").parse()
                    stages["rust_parse"] = True
                    retried = True
                except Exception:
                    pass
            if not retried:
                return OracleResult(
                    verdict=Verdict.ERROR.value,
                    error_msg=f"Rust parse failed: {e}",
                    time_ms=(time.time() - start) * 1000,
                    pipeline_stages=stages, func_name=name,
                )

        # Lower to IR
        try:
            c_module = CIRLowering().lower(c_ast)
            stages["c_ir"] = True
        except Exception as e:
            return OracleResult(
                verdict=Verdict.ERROR.value,
                error_msg=f"C IR lowering failed: {e}",
                time_ms=(time.time() - start) * 1000,
                pipeline_stages=stages, func_name=name,
            )

        try:
            r_module = RustIRLowering(RustTypeResolver()).lower(r_ast)
            stages["rust_ir"] = True
        except Exception as e:
            return OracleResult(
                verdict=Verdict.ERROR.value,
                error_msg=f"Rust IR lowering failed: {e}",
                time_ms=(time.time() - start) * 1000,
                pipeline_stages=stages, func_name=name,
            )

        # Align + product
        c_funcs = list(c_module.functions.values())
        r_funcs = list(r_module.functions.values())
        if not c_funcs or not r_funcs:
            return OracleResult(
                verdict=Verdict.ERROR.value,
                error_msg="No functions found in IR",
                time_ms=(time.time() - start) * 1000,
                pipeline_stages=stages, func_name=name,
            )

        c_func, r_func = c_funcs[0], r_funcs[0]

        try:
            alignment = FunctionAligner().align(c_func, r_func)
            stages["alignment"] = True
        except Exception as e:
            return OracleResult(
                verdict=Verdict.ERROR.value,
                error_msg=f"Alignment failed: {e}",
                time_ms=(time.time() - start) * 1000,
                pipeline_stages=stages, func_name=name,
            )

        c_config = SemanticConfig.c11()
        r_config = SemanticConfig.rust_release()

        try:
            product = ProductBuilder(c_config=c_config, rust_config=r_config).build(c_func, r_func)
            stages["product"] = True
        except Exception as e:
            return OracleResult(
                verdict=Verdict.ERROR.value,
                error_msg=f"Product build failed: {e}",
                time_ms=(time.time() - start) * 1000,
                pipeline_stages=stages, func_name=name,
            )

        # SMT verification
        try:
            verdict_str, cex_raw, n_queries = self._smt_verify(
                c_func, r_func, c_config, r_config, product
            )
            stages["smt"] = True
        except Exception as e:
            return OracleResult(
                verdict=Verdict.ERROR.value,
                error_msg=f"SMT verification failed: {e}",
                time_ms=(time.time() - start) * 1000,
                pipeline_stages=stages, func_name=name,
            )

        elapsed = (time.time() - start) * 1000

        # Build structured result
        cex_info = None
        hint = None
        if verdict_str == "divergent" and cex_raw:
            reason = cex_raw.get("reason", "semantic divergence")
            div_class = classify_divergence(reason)
            cex_info = CounterexampleInfo(
                inputs={k: v for k, v in cex_raw.items()
                        if k not in ("reason", "c_behavior", "rust_behavior")},
                reason=reason,
                divergence_class=div_class,
                c_behavior=cex_raw.get("c_behavior"),
                rust_behavior=cex_raw.get("rust_behavior"),
            )
            hint = generate_repair_hint(reason)

        return OracleResult(
            verdict=verdict_str,
            counterexample=cex_info,
            repair_hint=hint,
            time_ms=elapsed,
            smt_queries=n_queries,
            pipeline_stages=stages,
            func_name=name,
            confidence=1.0 if verdict_str in ("equivalent", "divergent") else 0.5,
        )

    def _smt_verify(self, c_func, r_func, c_config, r_config, product):
        """Run SMT verification (same logic as pipeline_verify.verify_product_program)."""
        import z3
        from src.smt.encoder import SMTEncoder, EncodingContext

        n_queries = 0
        c_args = list(c_func.arguments)
        r_args = list(r_func.arguments)
        min_args = min(len(c_args), len(r_args))

        shared_vars = []
        c_input_map = {}
        r_input_map = {}
        dummy_encoder = SMTEncoder(config=c_config)

        for i in range(min_args):
            sort = dummy_encoder.encode_type(c_args[i].type) if c_args[i].type else z3.BitVecSort(32)
            z3_var = z3.BitVec(f"input_{i}", sort.size()) if z3.is_bv_sort(sort) else z3.Const(f"input_{i}", sort)
            shared_vars.append((f"input_{i}", z3_var))
            ca_name = c_args[i].name or f"arg_{c_args[i].index}"
            ra_name = r_args[i].name or f"arg_{r_args[i].index}"
            c_input_map[ca_name] = z3_var
            r_input_map[ra_name] = z3_var

        # Encode both functions
        c_ctx = EncodingContext()
        for nm, var in c_input_map.items():
            c_ctx.declarations[nm] = var
            c_ctx._alloca_values[nm] = var
        c_encoder = SMTEncoder(config=c_config)
        _, c_ret = c_encoder.encode_function(c_func, c_ctx)

        r_ctx = EncodingContext()
        for nm, var in r_input_map.items():
            r_ctx.declarations[nm] = var
            r_ctx._alloca_values[nm] = var
        r_encoder = SMTEncoder(config=r_config)
        _, r_ret = r_encoder.encode_function(r_func, r_ctx)

        if c_ret is None or r_ret is None:
            return "unknown", None, 0

        solver = z3.Solver()
        solver.set("timeout", self.timeout_ms)
        for a in c_ctx.assumptions:
            solver.add(a)
        for a in c_ctx.assertions:
            solver.add(a)
        for a in r_ctx.assertions:
            solver.add(a)

        # Coerce return types
        c_r, r_r = c_ret, r_ret
        try:
            if z3.is_bv(c_r) and z3.is_bv(r_r):
                c_w, r_w = c_r.size(), r_r.size()
                if c_w != r_w:
                    if c_w < r_w:
                        c_r = z3.SignExt(r_w - c_w, c_r)
                    else:
                        r_r = z3.SignExt(c_w - r_w, r_r)
            elif z3.is_bool(c_r) and z3.is_bv(r_r):
                c_r = z3.If(c_r, z3.BitVecVal(1, r_r.size()), z3.BitVecVal(0, r_r.size()))
            elif z3.is_bv(c_r) and z3.is_bool(r_r):
                r_r = z3.If(r_r, z3.BitVecVal(1, c_r.size()), z3.BitVecVal(0, c_r.size()))
            solver.add(c_r != r_r)
        except Exception:
            return "unknown", None, 0

        def _model_val(model, var):
            val = model.evaluate(var, model_completion=True)
            if hasattr(val, 'as_signed_long'):
                return str(val.as_signed_long())
            if hasattr(val, 'as_long'):
                return str(val.as_long())
            return str(val)

        result = solver.check()
        n_queries += 1

        if result == z3.sat:
            m = solver.model()
            cex = {nm: _model_val(m, var) for nm, var in shared_vars}
            cex["reason"] = "output_mismatch"
            return "divergent", cex, n_queries
        elif result != z3.unsat:
            return "unknown", None, n_queries

        # Check UB divergence — distinguish output mismatch from UB-only divergence
        ub_divergence = False
        if c_ctx.assumptions:
            shared_ids = {var.get_id() for _, var in shared_vars}

            def _grounded(expr):
                if z3.is_const(expr):
                    return expr.get_id() in shared_ids or expr.num_args() == 0
                return all(_grounded(expr.arg(i)) for i in range(expr.num_args()))

            for i, assumption in enumerate(c_ctx.assumptions):
                if not _grounded(assumption):
                    continue
                s2 = z3.Solver()
                s2.set("timeout", self.timeout_ms)
                s2.add(z3.Not(assumption))
                check = s2.check()
                n_queries += 1
                if check == z3.sat:
                    ub_divergence = True
                    m = s2.model()
                    cex = {nm: _model_val(m, var) for nm, var in shared_vars}
                    cex["reason"] = f"c_undefined_behavior (assumption {i} violated)"
                    # In CEGAR mode: outputs match on well-defined inputs,
                    # only C UB inputs cause divergence. This means the Rust
                    # translation is correct (uses wrapping semantics).
                    if self._cegar_mode:
                        return "conditionally_equivalent", cex, n_queries
                    return "divergent", cex, n_queries

        return "equivalent", None, n_queries
