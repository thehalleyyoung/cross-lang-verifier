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
    LIKELY_EQUIVALENT = "likely_equivalent"
    LIKELY_DIVERGENT = "likely_divergent"
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
        return "overflow"
    if "null" in r or "nullptr" in r:
        return "null_pointer"
    if "pointer" in r or "provenance" in r or "dereference" in r:
        return "pointer"
    if "struct" in r or "layout" in r or "padding" in r or "alignment" in r:
        return "struct_layout"
    if "enum" in r or "discriminant" in r or "variant" in r:
        return "enum_discriminant"
    if "malloc" in r or "free" in r or "alloc" in r or "heap" in r:
        return "memory_management"
    if "lifetime" in r or "dangling" in r or "use_after_free" in r or "borrow" in r:
        return "lifetime"
    if "slice" in r or "bounds" in r or "array" in r or "index" in r:
        return "bounds_check"
    if "function_pointer" in r or "indirect_call" in r:
        return "function_pointer"
    if "string" in r or "utf" in r or "encoding" in r:
        return "string_encoding"
    if "union" in r or "type_pun" in r or "reinterpret" in r:
        return "union_reinterpret"
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
        import re
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

        # Parse C — prefer tree-sitter, fall back to hand-written, then try wrapping
        c_ast = None
        try:
            try:
                from src.frontend_c.tree_sitter_parser import TreeSitterCParser
                c_ast = TreeSitterCParser(c_code, f"{name}.c").parse()
                stages["c_parse"] = True
                stages["c_parser_backend"] = "tree-sitter"
            except Exception:
                c_ast = CParser(c_code, f"{name}.c").parse()
                stages["c_parse"] = True
                stages["c_parser_backend"] = "hand-written"
        except Exception as e:
            # Try wrapping in a function if it looks like statements
            for wrapper in [
                f"int {name}(void) {{ {c_code} }}",
                f"void {name}(void) {{ {c_code} }}",
            ]:
                try:
                    c_ast = CParser(wrapper, f"{name}.c").parse()
                    stages["c_parse"] = True
                    stages["c_parser_backend"] = "hand-written+wrapped"
                    break
                except Exception:
                    continue
            if c_ast is None:
                return OracleResult(
                    verdict=Verdict.ERROR.value,
                    error_msg=f"C parse failed: {e}",
                    time_ms=(time.time() - start) * 1000,
                    pipeline_stages=stages, func_name=name,
                )

        # Parse Rust — prefer tree-sitter, fall back with recovery strategies
        r_ast = None
        try:
            try:
                from src.frontend_rust.tree_sitter_parser import TreeSitterRustParser
                r_ast = TreeSitterRustParser(rust_code, f"{name}.rs").parse()
                stages["rust_parse"] = True
                stages["rust_parser_backend"] = "tree-sitter"
            except Exception:
                r_ast = RustParser(rust_code, f"{name}.rs").parse()
                stages["rust_parse"] = True
                stages["rust_parser_backend"] = "hand-written"
        except Exception as e:
            # Try recovery strategies
            variants = []
            if re.match(r'\s*fn\s+', rust_code):
                variants.append("pub " + rust_code)
            if not re.search(r'\bfn\s+', rust_code):
                variants.append(f"pub fn {name}() {{ {rust_code} }}")
            variants.append(f"pub {rust_code}")
            for v in variants:
                try:
                    r_ast = RustParser(v, f"{name}.rs").parse()
                    stages["rust_parse"] = True
                    stages["rust_parser_backend"] = "hand-written+recovery"
                    break
                except Exception:
                    continue
            if r_ast is None:
                return OracleResult(
                    verdict=Verdict.ERROR.value,
                    error_msg=f"Rust parse failed: {e}",
                    time_ms=(time.time() - start) * 1000,
                    pipeline_stages=stages, func_name=name,
                )

        # Lower to IR with recovery
        c_module = None
        try:
            c_module = CIRLowering().lower(c_ast)
            stages["c_ir"] = True
        except Exception as e:
            stages["c_ir"] = False
            stages["c_ir_error"] = str(e)

        r_module = None
        try:
            r_module = RustIRLowering(RustTypeResolver()).lower(r_ast)
            stages["rust_ir"] = True
        except Exception as e:
            stages["rust_ir"] = False
            stages["rust_ir_error"] = str(e)

        if c_module is None or r_module is None:
            # If both fail, return error; if one succeeds, try structural
            err_parts = []
            if c_module is None:
                err_parts.append(f"C IR: {stages.get('c_ir_error', 'unknown')}")
            if r_module is None:
                err_parts.append(f"Rust IR: {stages.get('rust_ir_error', 'unknown')}")
            return OracleResult(
                verdict=Verdict.ERROR.value,
                error_msg=f"IR lowering failed: {'; '.join(err_parts)}",
                time_ms=(time.time() - start) * 1000,
                pipeline_stages=stages, func_name=name,
            )

        # Find matching function pair (try all pairs if first fails)
        c_funcs = list(c_module.functions.values())
        r_funcs = list(r_module.functions.values())
        if not c_funcs or not r_funcs:
            return OracleResult(
                verdict=Verdict.ERROR.value,
                error_msg="No functions found in IR",
                time_ms=(time.time() - start) * 1000,
                pipeline_stages=stages, func_name=name,
            )

        # Try to find matching pair by name first, then fall back to first pair
        c_func, r_func = c_funcs[0], r_funcs[0]
        if len(c_funcs) > 1 or len(r_funcs) > 1:
            for cf in c_funcs:
                for rf in r_funcs:
                    if cf.name == rf.name:
                        c_func, r_func = cf, rf
                        break

        # Alignment with structural fallback
        alignment = None
        try:
            alignment = FunctionAligner().align(c_func, r_func)
            stages["alignment"] = True
        except Exception as e:
            stages["alignment"] = False
            stages["alignment_error"] = str(e)

        c_config = SemanticConfig.c11()
        r_config = SemanticConfig.rust_release()

        # Product construction with direct SMT fallback
        product = None
        if alignment is not None:
            try:
                product = ProductBuilder(c_config=c_config, rust_config=r_config).build(c_func, r_func)
                stages["product"] = True
            except Exception as e:
                stages["product"] = False
                stages["product_error"] = str(e)

        # SMT verification — try via product first, fall back to direct comparison
        verdict_str = "unknown"
        cex_raw = None
        n_queries = 0

        if product is not None:
            try:
                verdict_str, cex_raw, n_queries = self._smt_verify(
                    c_func, r_func, c_config, r_config, product
                )
                stages["smt"] = True
            except Exception as e:
                stages["smt"] = False
                stages["smt_error"] = str(e)

        # Direct SMT fallback when product/alignment failed
        if verdict_str == "unknown" and (product is None or not stages.get("smt")):
            try:
                verdict_str, cex_raw, dq = self._direct_smt_verify(
                    c_func, r_func, c_config, r_config
                )
                n_queries += dq
                stages["direct_smt"] = True
            except Exception as e:
                stages["direct_smt"] = False
                stages["direct_smt_error"] = str(e)

        # Structural verification as additional signal
        structural_verdict = None
        try:
            structural_verdict = self._structural_verify(c_func, r_func)
            stages["structural"] = True
            stages["structural_verdict"] = structural_verdict
        except Exception:
            stages["structural"] = False

        # Enhanced memory model analysis when available
        try:
            from src.smt.points_to_analysis import EnhancedMemoryModel
            from src.smt.encoder import EncodingContext as EMCtx
            emm = EnhancedMemoryModel(EMCtx(), c_config, r_config)
            emm.analyze_c_function(c_func)
            emm.analyze_rust_function(r_func)
            mem_stats = emm.encode_all_constraints()
            stages["enhanced_memory"] = True
            stages["memory_stats"] = mem_stats
        except Exception:
            stages["enhanced_memory"] = False

        elapsed = (time.time() - start) * 1000

        # Refine verdict using structural info
        confidence = 1.0
        if verdict_str == "equivalent":
            confidence = 0.95 if n_queries <= 1 else 1.0
            if structural_verdict and structural_verdict.get("similar") is False:
                verdict_str = "likely_equivalent"
                confidence = 0.7
        elif verdict_str == "divergent":
            confidence = 1.0
        elif verdict_str == "unknown":
            if structural_verdict:
                sim = structural_verdict.get("similarity", 0)
                if sim >= 0.95:
                    verdict_str = "likely_equivalent"
                    confidence = 0.6
                elif sim < 0.3:
                    verdict_str = "likely_divergent"
                    confidence = 0.4
                else:
                    confidence = sim * 0.5
            else:
                confidence = 0.0

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
            confidence=confidence,
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
        r_dummy_encoder = SMTEncoder(config=r_config)

        for i in range(min_args):
            # Prefer C arg type, fall back to Rust, fall back to i32
            sort = z3.BitVecSort(32)
            try:
                if c_args[i].type:
                    sort = dummy_encoder.encode_type(c_args[i].type)
            except Exception:
                try:
                    if r_args[i].type:
                        sort = r_dummy_encoder.encode_type(r_args[i].type)
                except Exception:
                    pass
            try:
                z3_var = z3.BitVec(f"input_{i}", sort.size()) if z3.is_bv_sort(sort) else z3.Const(f"input_{i}", sort)
            except Exception:
                z3_var = z3.BitVec(f"input_{i}", 32)
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
                    target_w = max(c_w, r_w)
                    if c_w < target_w:
                        c_r = z3.SignExt(target_w - c_w, c_r)
                    if r_w < target_w:
                        r_r = z3.SignExt(target_w - r_w, r_r)
            elif z3.is_bool(c_r) and z3.is_bv(r_r):
                c_r = z3.If(c_r, z3.BitVecVal(1, r_r.size()), z3.BitVecVal(0, r_r.size()))
            elif z3.is_bv(c_r) and z3.is_bool(r_r):
                r_r = z3.If(r_r, z3.BitVecVal(1, c_r.size()), z3.BitVecVal(0, c_r.size()))
            elif z3.is_bool(c_r) and z3.is_bool(r_r):
                pass  # Both bool, can compare directly
            elif z3.is_int(c_r) and z3.is_bv(r_r):
                c_r = z3.Int2BV(c_r, r_r.size())
            elif z3.is_bv(c_r) and z3.is_int(r_r):
                r_r = z3.Int2BV(r_r, c_r.size())
            elif z3.is_fp(c_r) and z3.is_fp(r_r):
                pass  # Both FP, can compare directly
            else:
                # Try to convert both to 32-bit bitvectors
                if not z3.is_bv(c_r):
                    c_r = z3.BitVecVal(0, 32)
                if not z3.is_bv(r_r):
                    r_r = z3.BitVecVal(0, 32)
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
            # Check if counterexample triggers C undefined behavior
            # If all UB assumptions hold under this model, it's a genuine divergence
            # If any UB assumption is violated, the C behavior is undefined,
            # so this divergence is on a UB input — try constraining to well-defined inputs
            m = solver.model()
            cex = {nm: _model_val(m, var) for nm, var in shared_vars}

            ub_violated = False
            if c_ctx.assumptions:
                for assumption in c_ctx.assumptions:
                    try:
                        val = m.evaluate(assumption, model_completion=True)
                        if z3.is_false(val):
                            ub_violated = True
                            break
                    except Exception:
                        pass

            if ub_violated:
                # The counterexample triggers C UB — try again with UB preconditions
                solver2 = z3.Solver()
                solver2.set("timeout", self.timeout_ms)
                for a in c_ctx.assumptions:
                    solver2.add(a)
                for a in c_ctx.assertions:
                    solver2.add(a)
                for a in r_ctx.assertions:
                    solver2.add(a)
                solver2.add(c_r != r_r)
                result2 = solver2.check()
                n_queries += 1

                if result2 == z3.sat:
                    m2 = solver2.model()
                    cex2 = {nm: _model_val(m2, var) for nm, var in shared_vars}
                    cex2["reason"] = "output_mismatch (on well-defined inputs)"
                    return "divergent", cex2, n_queries
                elif result2 == z3.unsat:
                    # Equivalent on all well-defined inputs
                    # But diverges on UB inputs — report as equivalent
                    cex["reason"] = "c_undefined_behavior (outputs match on well-defined inputs)"
                    if self._cegar_mode:
                        return "conditionally_equivalent", cex, n_queries
                    return "equivalent", None, n_queries
                else:
                    return "unknown", None, n_queries

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

    def _direct_smt_verify(self, c_func, r_func, c_config, r_config):
        """Direct SMT comparison without product program."""
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
            sort = z3.BitVecSort(32)
            try:
                if c_args[i].type:
                    sort = dummy_encoder.encode_type(c_args[i].type)
            except Exception:
                pass
            try:
                z3_var = z3.BitVec(f"input_{i}", sort.size()) if z3.is_bv_sort(sort) \
                    else z3.Const(f"input_{i}", sort)
            except Exception:
                z3_var = z3.BitVec(f"input_{i}", 32)
            shared_vars.append((f"input_{i}", z3_var))
            ca_name = c_args[i].name or f"arg_{c_args[i].index}"
            ra_name = r_args[i].name or f"arg_{r_args[i].index}"
            c_input_map[ca_name] = z3_var
            r_input_map[ra_name] = z3_var

        c_ctx = EncodingContext()
        for nm, var in c_input_map.items():
            c_ctx.declarations[nm] = var
            c_ctx._alloca_values[nm] = var
        _, c_ret = SMTEncoder(config=c_config).encode_function(c_func, c_ctx)

        r_ctx = EncodingContext()
        for nm, var in r_input_map.items():
            r_ctx.declarations[nm] = var
            r_ctx._alloca_values[nm] = var
        _, r_ret = SMTEncoder(config=r_config).encode_function(r_func, r_ctx)

        if c_ret is None or r_ret is None:
            return "unknown", None, 0

        solver = z3.Solver()
        solver.set("timeout", self.timeout_ms)
        for a in c_ctx.assertions:
            solver.add(a)
        for a in r_ctx.assertions:
            solver.add(a)

        c_r, r_r = c_ret, r_ret
        try:
            if z3.is_bv(c_r) and z3.is_bv(r_r):
                c_w, r_w = c_r.size(), r_r.size()
                if c_w != r_w:
                    tw = max(c_w, r_w)
                    if c_w < tw:
                        c_r = z3.SignExt(tw - c_w, c_r)
                    if r_w < tw:
                        r_r = z3.SignExt(tw - r_w, r_r)
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
        elif result == z3.unsat:
            return "equivalent", None, n_queries
        return "unknown", None, n_queries

    def _structural_verify(self, c_func, r_func) -> dict:
        """Compare function signatures, instruction counts, and control flow
        as a fast heuristic. Returns a dict with similarity metrics."""
        c_instr = c_func.instruction_count
        r_instr = r_func.instruction_count
        c_blocks = c_func.num_blocks
        r_blocks = r_func.num_blocks

        # Opcode histograms
        c_ops: dict = {}
        for inst in c_func.iter_instructions():
            op = inst.opcode_name()
            c_ops[op] = c_ops.get(op, 0) + 1
        r_ops: dict = {}
        for inst in r_func.iter_instructions():
            op = inst.opcode_name()
            r_ops[op] = r_ops.get(op, 0) + 1

        all_ops = set(c_ops) | set(r_ops)
        if all_ops:
            matching = sum(min(c_ops.get(o, 0), r_ops.get(o, 0)) for o in all_ops)
            total = sum(max(c_ops.get(o, 0), r_ops.get(o, 0)) for o in all_ops)
            opcode_sim = matching / total if total else 0.0
        else:
            opcode_sim = 1.0

        max_instr = max(c_instr, r_instr, 1)
        instr_sim = 1.0 - abs(c_instr - r_instr) / max_instr

        # Return type compatibility
        ret_compat = 1.0
        try:
            c_ret = c_func.return_type
            r_ret = r_func.return_type
            if c_ret != r_ret:
                ret_compat = 0.5
            c_void = getattr(c_ret, 'is_void', False)
            r_void = getattr(r_ret, 'is_void', False)
            if c_void != r_void:
                ret_compat = 0.0
        except Exception:
            ret_compat = 0.5

        # Argument count compatibility
        c_nargs = len(list(c_func.arguments))
        r_nargs = len(list(r_func.arguments))
        arg_compat = 1.0 if c_nargs == r_nargs else max(0.0, 1.0 - abs(c_nargs - r_nargs) * 0.25)

        similarity = (0.35 * opcode_sim + 0.3 * instr_sim +
                      0.15 * ret_compat + 0.2 * arg_compat)

        return {
            "similarity": similarity,
            "similar": similarity >= 0.8,
            "opcode_similarity": opcode_sim,
            "instruction_similarity": instr_sim,
            "return_compatible": ret_compat > 0.0,
            "arg_count_match": c_nargs == r_nargs,
            "c_blocks": c_blocks,
            "r_blocks": r_blocks,
        }
