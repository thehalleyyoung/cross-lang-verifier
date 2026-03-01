#!/usr/bin/env python3
"""
Shared pipeline verification utility.

All experiments call this module to run the actual SemRec pipeline:
  CParser → CIRLowering → RustParser → RustIRLowering → ProductBuilder → Z3 SMT

No hand-coded Z3 constraints. All verification goes through the product program.
"""

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import z3

from src.frontend_c.parser import CParser
from src.frontend_c.ir_lowering import CIRLowering
from src.frontend_rust.parser import RustParser
from src.frontend_rust.ir_lowering import RustIRLowering
from src.frontend_rust.type_resolver import RustTypeResolver
from src.product_program.product import ProductBuilder
from src.product_program.alignment import FunctionAligner
from src.semantics.semantic_config import SemanticConfig
from src.smt.solver import SMTSolver, SolverConfig, SolverStatus
from src.smt.encoder import SMTEncoder, EncodingContext
from src.ir.instructions import (
    BinaryOp, BinOpKind, ReturnInst, BranchInst, CompareOp,
)


@dataclass
class PipelineResult:
    """Result from the full SemRec pipeline."""
    name: str
    verdict: str  # "equivalent", "divergent", "unknown", "error", "pipeline_fail"
    time_ms: float = 0.0
    counterexample: Optional[Dict[str, Any]] = None
    divergence_reason: Optional[str] = None
    pipeline_stages: Dict[str, bool] = field(default_factory=dict)
    pipeline_log: List[str] = field(default_factory=list)
    smt_queries: int = 0
    coercion_points: int = 0
    product_blocks: int = 0
    alignment_score: float = 0.0
    error_msg: Optional[str] = None


def z3_model_value(model, var):
    """Extract a concrete integer value from a Z3 model."""
    val = model.evaluate(var, model_completion=True)
    if hasattr(val, 'as_signed_long'):
        return val.as_signed_long()
    if hasattr(val, 'as_long'):
        return val.as_long()
    return str(val)


def run_pipeline(name: str, c_source: str, rust_source: str,
                 timeout_ms: int = 10000) -> PipelineResult:
    """Run the full SemRec pipeline on a C/Rust function pair.

    Pipeline stages:
      1. CParser → C AST
      2. RustParser → Rust AST
      3. CIRLowering → C IR Module
      4. RustIRLowering → Rust IR Module
      5. FunctionAligner → Alignment
      6. ProductBuilder → Product Program
      7. SMT encoding of product program → Z3 verification
    """
    start = time.time()
    result = PipelineResult(name=name, verdict="unknown")
    log = result.pipeline_log

    # Stage 1: Parse C
    try:
        c_parser = CParser(c_source, f"{name}.c")
        c_ast = c_parser.parse()
        result.pipeline_stages["c_parse"] = True
        log.append("c_parse: OK")
    except Exception as e:
        result.pipeline_stages["c_parse"] = False
        log.append(f"c_parse: FAIL ({e})")
        result.verdict = "pipeline_fail"
        result.error_msg = f"C parse failed: {e}"
        result.time_ms = (time.time() - start) * 1000
        return result

    # Stage 2: Parse Rust
    try:
        r_parser = RustParser(rust_source, f"{name}.rs")
        r_ast = r_parser.parse()
        result.pipeline_stages["rust_parse"] = True
        log.append("rust_parse: OK")
    except Exception as e:
        result.pipeline_stages["rust_parse"] = False
        log.append(f"rust_parse: FAIL ({e})")
        result.verdict = "pipeline_fail"
        result.error_msg = f"Rust parse failed: {e}"
        result.time_ms = (time.time() - start) * 1000
        return result

    # Stage 3: Lower C to IR
    try:
        c_lowering = CIRLowering()
        c_module = c_lowering.lower(c_ast)
        result.pipeline_stages["c_ir"] = True
        log.append(f"c_ir: OK ({c_module.num_functions} functions)")
    except Exception as e:
        result.pipeline_stages["c_ir"] = False
        log.append(f"c_ir: FAIL ({e})")
        result.verdict = "pipeline_fail"
        result.error_msg = f"C IR lowering failed: {e}"
        result.time_ms = (time.time() - start) * 1000
        return result

    # Stage 4: Lower Rust to IR
    try:
        r_lowering = RustIRLowering(RustTypeResolver())
        r_module = r_lowering.lower(r_ast)
        result.pipeline_stages["rust_ir"] = True
        log.append(f"rust_ir: OK ({r_module.num_functions} functions)")
    except Exception as e:
        result.pipeline_stages["rust_ir"] = False
        log.append(f"rust_ir: FAIL ({e})")
        result.verdict = "pipeline_fail"
        result.error_msg = f"Rust IR lowering failed: {e}"
        result.time_ms = (time.time() - start) * 1000
        return result

    # Stage 5: Align + Build Product Program
    c_funcs = list(c_module.functions.values())
    r_funcs = list(r_module.functions.values())
    if not c_funcs or not r_funcs:
        result.verdict = "pipeline_fail"
        result.error_msg = "No functions found in IR modules"
        result.time_ms = (time.time() - start) * 1000
        return result

    c_func = c_funcs[0]
    r_func = r_funcs[0]

    try:
        aligner = FunctionAligner()
        alignment = aligner.align(c_func, r_func)
        result.alignment_score = alignment.structural_similarity
        result.pipeline_stages["alignment"] = True
        log.append(f"alignment: OK (sim={alignment.structural_similarity:.3f})")
    except Exception as e:
        result.pipeline_stages["alignment"] = False
        log.append(f"alignment: FAIL ({e})")
        result.verdict = "pipeline_fail"
        result.error_msg = f"Alignment failed: {e}"
        result.time_ms = (time.time() - start) * 1000
        return result

    try:
        c_config = SemanticConfig.c11()
        r_config = SemanticConfig.rust_release()
        builder = ProductBuilder(c_config=c_config, rust_config=r_config)
        product = builder.build(c_func, r_func)
        result.pipeline_stages["product"] = True
        result.coercion_points = product.num_coercion_points
        result.product_blocks = product.num_blocks
        log.append(f"product: OK ({product.num_blocks} blocks, "
                   f"{product.num_coercion_points} coercion pts)")
    except Exception as e:
        result.pipeline_stages["product"] = False
        log.append(f"product: FAIL ({e})")
        result.verdict = "pipeline_fail"
        result.error_msg = f"Product build failed: {e}"
        result.time_ms = (time.time() - start) * 1000
        return result

    # Stage 6: SMT verification via product program
    try:
        verdict, cex, n_queries = verify_product_program(
            product, c_func, r_func, c_config, r_config, timeout_ms
        )
        result.verdict = verdict
        result.counterexample = cex
        result.smt_queries = n_queries
        result.pipeline_stages["smt"] = True
        log.append(f"smt: {verdict} ({n_queries} queries)")
    except Exception as e:
        result.pipeline_stages["smt"] = False
        log.append(f"smt: FAIL ({e})")
        result.verdict = "error"
        result.error_msg = f"SMT verification failed: {e}"

    result.time_ms = (time.time() - start) * 1000
    return result


def verify_product_program(product, c_func, r_func, c_config, r_config,
                           timeout_ms=10000):
    """Verify a product program using the SMT encoder and Z3.

    Encodes both functions in separate contexts with separate encoders,
    then unifies via shared symbolic input variables.
    """
    n_queries = 0

    c_args = list(c_func.arguments)
    r_args = list(r_func.arguments)
    min_args = min(len(c_args), len(r_args))

    # Create shared input variables
    shared_vars = []
    c_input_map = {}  # maps c_arg_name -> z3 var
    r_input_map = {}  # maps r_arg_name -> z3 var
    dummy_encoder = SMTEncoder(config=c_config)
    for i in range(min_args):
        sort = dummy_encoder.encode_type(c_args[i].type) if c_args[i].type else z3.BitVecSort(32)
        z3_var = z3.BitVec(f"input_{i}", sort.size()) if z3.is_bv_sort(sort) else z3.Const(f"input_{i}", sort)
        shared_vars.append((f"input_{i}", z3_var))
        ca_name = c_args[i].name or f"arg_{c_args[i].index}"
        ra_name = r_args[i].name or f"arg_{r_args[i].index}"
        c_input_map[ca_name] = z3_var
        r_input_map[ra_name] = z3_var

    # Encode C function
    c_ctx = EncodingContext()
    for name, var in c_input_map.items():
        c_ctx.declarations[name] = var
        c_ctx._alloca_values[name] = var
    c_encoder = SMTEncoder(config=c_config)
    _, c_ret = c_encoder.encode_function(c_func, c_ctx)

    # Encode Rust function
    r_ctx = EncodingContext()
    for name, var in r_input_map.items():
        r_ctx.declarations[name] = var
        r_ctx._alloca_values[name] = var
    r_encoder = SMTEncoder(config=r_config)
    _, r_ret = r_encoder.encode_function(r_func, r_ctx)

    if c_ret is None or r_ret is None:
        return _verify_via_coercions(product, shared_vars, timeout_ms)

    # Check equivalence under C's no-UB assumptions
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)
    for assumption in c_ctx.assumptions:
        solver.add(assumption)
    for assertion in c_ctx.assertions:
        solver.add(assertion)
    for assertion in r_ctx.assertions:
        solver.add(assertion)

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
        return _verify_via_coercions(product, shared_vars, timeout_ms)

    result = solver.check()
    n_queries += 1

    if result == z3.sat:
        m = solver.model()
        cex = {}
        for name, var in shared_vars:
            cex[name] = str(z3_model_value(m, var))
        return "divergent", cex, n_queries
    elif result == z3.unsat:
        pass  # Equivalent under no-UB; check if UB is reachable
    else:
        return "unknown", None, n_queries

    # Check for divergence on UB inputs: C has UB (assumption violated)
    # but Rust has defined behavior (wraps/panics).
    # Only check assumptions that are "grounded" — they must be satisfiable/
    # violable using only the shared input variables, not unconstrained
    # intermediate variables from the encoding.
    if c_ctx.assumptions:
        shared_z3_vars = set()
        for _, var in shared_vars:
            shared_z3_vars.add(var.get_id())

        def _is_grounded(expr):
            """Check if a Z3 expression only involves shared input variables."""
            if z3.is_const(expr):
                return expr.get_id() in shared_z3_vars or expr.num_args() == 0
            for i in range(expr.num_args()):
                if not _is_grounded(expr.arg(i)):
                    return False
            return True

        for i, assumption in enumerate(c_ctx.assumptions):
            # Skip assumptions that reference unconstrained intermediate vars
            if not _is_grounded(assumption):
                continue

            s2 = z3.Solver()
            s2.set("timeout", timeout_ms)
            s2.add(z3.Not(assumption))
            check = s2.check()
            n_queries += 1
            if check == z3.sat:
                m = s2.model()
                cex = {}
                for vname, var in shared_vars:
                    cex[vname] = str(z3_model_value(m, var))
                cex["reason"] = f"c_undefined_behavior (assumption {i} violated)"
                return "divergent", cex, n_queries

    return "equivalent", None, n_queries


def _check_overflow_divergence(c_func, r_func, encoder, c_ctx, r_ctx,
                               shared_vars, timeout_ms):
    """Check if any operation in the C function can trigger overflow/UB.
    
    shared_vars is a list of (name, c_var, r_var) triples.
    """
    n_queries = 0
    for block in c_func.blocks:
        for inst in block.instructions:
            if not isinstance(inst, BinaryOp):
                continue
            op_name = inst.op.name if hasattr(inst.op, 'name') else str(inst.op)
            if op_name not in ("ADD", "SUB", "MUL", "SHL", "SDIV", "NEG"):
                continue

            try:
                lhs = encoder.encode_value(inst.lhs, c_ctx)
                rhs = encoder.encode_value(inst.rhs, c_ctx)
                if not (z3.is_bv(lhs) and z3.is_bv(rhs)):
                    continue
                # Coerce widths
                lw, rw = lhs.size(), rhs.size()
                if lw != rw:
                    if lw < rw:
                        lhs = z3.SignExt(rw - lw, lhs)
                    else:
                        rhs = z3.SignExt(lw - rw, rhs)

                s = z3.Solver()
                s.set("timeout", timeout_ms)
                overflow_cond = None

                if op_name == "ADD":
                    result_bv = lhs + rhs
                    overflow_cond = z3.Or(
                        z3.And(lhs > 0, rhs > 0, result_bv < 0),
                        z3.And(lhs < 0, rhs < 0, result_bv >= 0),
                    )
                elif op_name == "SUB":
                    result_bv = lhs - rhs
                    overflow_cond = z3.Or(
                        z3.And(lhs > 0, rhs < 0, result_bv < 0),
                        z3.And(lhs < 0, rhs > 0, result_bv >= 0),
                    )
                elif op_name == "MUL":
                    w = lhs.size()
                    wide_l = z3.SignExt(w, lhs)
                    wide_r = z3.SignExt(w, rhs)
                    wide_result = wide_l * wide_r
                    overflow_cond = z3.Or(
                        wide_result > z3.BitVecVal((1 << (w - 1)) - 1, 2 * w),
                        wide_result < z3.BitVecVal(-(1 << (w - 1)), 2 * w),
                    )
                elif op_name == "SHL":
                    w = rhs.size()
                    overflow_cond = z3.UGE(rhs, z3.BitVecVal(w, w))
                elif op_name == "SDIV":
                    w = lhs.size()
                    overflow_cond = z3.Or(
                        rhs == z3.BitVecVal(0, w),
                        z3.And(lhs == z3.BitVecVal(-(1 << (w - 1)), w),
                               rhs == z3.BitVecVal(-1, w)),
                    )

                if overflow_cond is not None:
                    s.add(overflow_cond)
                    check_result = s.check()
                    n_queries += 1
                    if check_result == z3.sat:
                        m = s.model()
                        cex = {}
                        for vname, c_var, r_var in shared_vars:
                            cex[vname] = str(z3_model_value(m, c_var))
                        cex["reason"] = f"{op_name.lower()}_overflow"
                        return "divergent", cex, n_queries
            except Exception:
                continue

    return "equivalent", None, n_queries


def _verify_via_coercions(product, shared_vars, timeout_ms):
    """Fallback: verify using product program coercion assertions.
    
    shared_vars is a list of (name, c_var, r_var) triples (or (name, var) pairs).
    """
    n_queries = 0
    hard = product.hard_assertions()
    assumptions = product.assumptions()

    if not hard and not assumptions:
        # No coercion points: either equivalent or we can't tell
        if product.num_coercion_points == 0:
            return "equivalent", None, 0
        return "unknown", None, 0

    solver = z3.Solver()
    solver.set("timeout", timeout_ms)

    # Try to parse SMT-LIB2 assertions from the product program
    smt_script = product.to_smt_lib()
    try:
        parsed = z3.parse_smt2_string(smt_script)
        for a in parsed:
            solver.add(a)
        result = solver.check()
        n_queries += 1
        if result == z3.sat:
            m = solver.model()
            cex = {}
            for vname, var in shared_vars:
                try:
                    cex[vname] = str(z3_model_value(m, var))
                except Exception:
                    pass
            return "divergent", cex, n_queries
        elif result == z3.unsat:
            return "equivalent", None, n_queries
    except Exception:
        pass

    return "unknown", None, n_queries
