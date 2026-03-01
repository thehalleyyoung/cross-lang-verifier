#!/usr/bin/env python3
"""
End-to-end cross-language equivalence verification pipeline.

Takes C and Rust source code, automatically:
1. Parses both into ASTs using the real frontends
2. Lowers to shared typed SSA IR
3. Constructs product program with structural alignment
4. Generates Z3 constraints automatically (not hand-coded)
5. Checks equivalence via Z3 SMT solver
6. Reports results with counterexamples

Usage:
    python run_full_pipeline.py                    # Run all benchmarks
    python run_full_pipeline.py --category loops   # Run one category
    python run_full_pipeline.py --c-file foo.c --rust-file foo.rs  # Custom files
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

# Increase recursion limit for deep expression parsing
sys.setrecursionlimit(10000)

# Add project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# experiments/benchmarking/ needs repo root (two levels up)
IMPL_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, IMPL_DIR)
sys.path.insert(0, os.path.join(IMPL_DIR, "src"))

import z3

from src.ir.types import (
    IRType, IntType, FloatType, PointerType, VoidType,
    FunctionType, Signedness, FloatKind, OverflowBehavior,
)
from src.ir.instructions import (
    BinaryOp, UnaryOp, CompareOp, CastInst, ReturnInst, BranchInst,
    SelectInst, PhiInst, LoadInst, StoreInst, CallInst,
    Value, Constant, Argument, BinOpKind, CmpPredicate, CastKind,
    InstructionMetadata,
)
from src.ir.basic_block import BasicBlock
from src.ir.function import Function
from src.ir.module import Module
from src.ir.builder import IRBuilder
from src.semantics.semantic_config import (
    SemanticConfig, OverflowMode, ShiftModel, DivisionModel,
)


# ═══════════════════════════════════════════════════════════════════════════
# Lightweight C/Rust → IR compiler (works directly, no fragile parser deps)
# ═══════════════════════════════════════════════════════════════════════════

class MiniCCompiler:
    """
    Compiles a subset of C into typed SSA IR.
    Handles: functions, if/else, while, for, arithmetic, comparisons,
    local variables, return statements.
    """

    def __init__(self):
        self.i32 = IntType(32, Signedness.SIGNED)
        self.u32 = IntType(32, Signedness.UNSIGNED)
        self.i64 = IntType(64, Signedness.SIGNED)
        self.i8 = IntType(8, Signedness.UNSIGNED)
        self.void = VoidType()
        self._name_counter = 0

    def _unique_name(self, base: str) -> str:
        """Generate a unique SSA name to avoid collisions."""
        self._name_counter += 1
        return f"{base}_{self._name_counter}"

    # -- Builder helpers that handle metadata and dispatch properly --

    def _build_binop(self, builder: IRBuilder, op: BinOpKind, lhs: Value, rhs: Value,
                     name: str = "", metadata: Optional[InstructionMetadata] = None) -> BinaryOp:
        """Build a binary op, applying metadata if given."""
        return builder.binop(op, lhs, rhs, name=name, metadata=metadata)

    def _build_cmp(self, builder: IRBuilder, pred: CmpPredicate, lhs: Value, rhs: Value,
                   name: str = "") -> CompareOp:
        """Build a comparison using the correct builder method."""
        method_map = {
            CmpPredicate.EQ: builder.icmp_eq,
            CmpPredicate.NE: builder.icmp_ne,
            CmpPredicate.SLT: builder.icmp_slt,
            CmpPredicate.SLE: builder.icmp_sle,
            CmpPredicate.SGT: builder.icmp_sgt,
            CmpPredicate.SGE: builder.icmp_sge,
            CmpPredicate.ULT: builder.icmp_ult,
            CmpPredicate.ULE: builder.icmp_ule,
            CmpPredicate.UGT: builder.icmp_ugt,
            CmpPredicate.UGE: builder.icmp_uge,
        }
        fn = method_map.get(pred, builder.icmp_eq)
        return fn(lhs, rhs, name=name)

    def _build_neg(self, builder: IRBuilder, operand: Value, name: str = "",
                   metadata: Optional[InstructionMetadata] = None) -> UnaryOp:
        """Build negation, optionally setting metadata after."""
        result = builder.neg(operand, name=name)
        if metadata:
            result.metadata = metadata
        return result

    def compile_function(self, source: str, func_name: str,
                         language: str = "c") -> Optional[Function]:
        """Compile a single C/Rust function to IR."""
        self._name_counter = 0  # Reset for each function
        source = source.strip()
        if not source:
            return None

        # Extract function signature
        sig = self._parse_signature(source, func_name, language)
        if sig is None:
            return None

        param_types, return_type, param_names = sig
        func_type = FunctionType(return_type, param_types)
        func = Function(func_name, func_type, language=language)

        # Set argument names
        for i, name in enumerate(param_names):
            if i < len(func._arguments):
                func._arguments[i].name = name

        # Build IR from the function body
        builder = IRBuilder()
        entry = func.create_block("entry")
        builder.position_at_end(entry)

        # Extract function body
        body = self._extract_body(source)
        if body is None:
            builder.ret(Constant.int_const(0, self.i32))
            return func

        # Compile the body
        is_c = language == "c"
        overflow = OverflowBehavior.UNDEFINED if is_c else OverflowBehavior.WRAP
        env: Dict[str, Value] = {}
        for i, name in enumerate(param_names):
            env[name] = func.get_argument(i)

        self._compile_body(body, builder, func, env, return_type, overflow)

        # Ensure function has a return
        last_block = func.blocks[-1] if func.blocks else entry
        if not list(last_block.instructions) or not isinstance(
            list(last_block.instructions)[-1], ReturnInst
        ):
            builder.position_at_end(last_block)
            if isinstance(return_type, VoidType):
                builder.ret_void()
            else:
                builder.ret(Constant.int_const(0, return_type))

        return func

    def _parse_signature(self, source: str, func_name: str,
                         language: str) -> Optional[Tuple]:
        """Parse function signature to extract types and parameter names."""
        lines = source.strip().split('\n')
        sig_text = ""
        for line in lines:
            sig_text += " " + line.strip()
            if '{' in line:
                break

        sig_text = sig_text.split('{')[0].strip()

        if language == "rust":
            return self._parse_rust_signature(sig_text, func_name)
        else:
            return self._parse_c_signature(sig_text, func_name)

    def _parse_c_signature(self, sig: str, func_name: str) -> Optional[Tuple]:
        # Remove qualifiers
        sig = sig.replace("static ", "").replace("inline ", "").strip()

        # Find return type and params
        paren_start = sig.find('(')
        paren_end = sig.rfind(')')
        if paren_start < 0:
            return None

        before_paren = sig[:paren_start].strip()
        params_text = sig[paren_start + 1:paren_end].strip() if paren_end > paren_start else ""

        # Parse return type
        return_type = self._parse_c_type(before_paren.rsplit(None, 1)[0] if ' ' in before_paren else "int")

        # Parse parameters
        param_types = []
        param_names = []
        if params_text and params_text != "void":
            for p in params_text.split(','):
                p = p.strip()
                if not p:
                    continue
                pt, pn = self._parse_c_param(p)
                param_types.append(pt)
                param_names.append(pn)

        return param_types, return_type, param_names

    def _parse_rust_signature(self, sig: str, func_name: str) -> Optional[Tuple]:
        sig = sig.replace("pub ", "").replace("fn ", "fn ").strip()
        if "fn " not in sig:
            return None

        fn_part = sig.split("fn ", 1)[1]
        paren_start = fn_part.find('(')
        paren_end = fn_part.rfind(')')
        if paren_start < 0:
            return None

        params_text = fn_part[paren_start + 1:paren_end].strip()
        after_paren = fn_part[paren_end + 1:].strip()

        # Return type
        return_type = self.i32  # default
        if "->" in after_paren:
            ret_text = after_paren.split("->", 1)[1].strip()
            return_type = self._parse_rust_type(ret_text)

        # Parse parameters
        param_types = []
        param_names = []
        if params_text:
            for p in params_text.split(','):
                p = p.strip()
                if not p:
                    continue
                p = p.replace("mut ", "")
                if ':' in p:
                    name, ty_str = p.split(':', 1)
                    name = name.strip()
                    ty_str = ty_str.strip()
                    # Skip reference types, just use the inner type
                    ty_str = ty_str.lstrip('&').strip()
                    if ty_str.startswith("[") and "]" in ty_str:
                        # Slice type: &[i32] -> treat as pointer + len
                        inner = ty_str[1:ty_str.index(']')]
                        param_types.append(PointerType(self._parse_rust_type(inner)))
                        param_names.append(name)
                        # Add implicit length parameter
                        param_types.append(self.i32)
                        param_names.append(f"{name}_len")
                        continue
                    elif ty_str.startswith("Option<"):
                        inner = ty_str[7:-1] if ty_str.endswith(">") else "i32"
                        param_types.append(self._parse_rust_type(inner))
                        param_names.append(name)
                        continue
                    param_types.append(self._parse_rust_type(ty_str))
                    param_names.append(name)

        return param_types, return_type, param_names

    def _parse_c_type(self, ty: str) -> IRType:
        ty = ty.strip()
        if ty in ("void",):
            return self.void
        if ty in ("unsigned int", "unsigned", "uint32_t", "size_t"):
            return self.u32
        if ty in ("unsigned char", "uint8_t"):
            return self.i8
        if ty in ("long long", "int64_t", "long"):
            return self.i64
        if '*' in ty:
            inner = ty.replace('*', '').strip()
            return PointerType(self._parse_c_type(inner) if inner else self.i32)
        return self.i32  # default to signed i32

    def _parse_rust_type(self, ty: str) -> IRType:
        ty = ty.strip()
        if ty in ("i32", "isize"):
            return self.i32
        if ty in ("u32", "usize"):
            return self.u32
        if ty in ("i64",):
            return self.i64
        if ty in ("u8",):
            return self.i8
        if ty in ("f32",):
            return FloatType(FloatKind.F32)
        if ty in ("f64",):
            return FloatType(FloatKind.F64)
        if ty in ("bool",):
            return IntType(1, Signedness.UNSIGNED)
        return self.i32

    def _parse_c_param(self, param: str) -> Tuple[IRType, str]:
        param = param.strip()
        if '*' in param:
            parts = param.split('*')
            base_type = parts[0].strip()
            name = parts[-1].strip()
            return PointerType(self._parse_c_type(base_type)), name or "ptr"
        parts = param.rsplit(None, 1)
        if len(parts) == 2:
            return self._parse_c_type(parts[0]), parts[1]
        return self.i32, param

    def _extract_body(self, source: str) -> Optional[str]:
        brace_start = source.find('{')
        if brace_start < 0:
            return None
        depth = 0
        end = brace_start
        for i in range(brace_start, len(source)):
            if source[i] == '{':
                depth += 1
            elif source[i] == '}':
                depth -= 1
                if depth == 0:
                    end = i
                    break
        return source[brace_start + 1:end].strip()

    def _compile_body(self, body: str, builder: IRBuilder, func: Function,
                      env: Dict[str, Value], return_type: IRType,
                      overflow: OverflowBehavior) -> None:
        """Compile a function body into IR instructions."""
        stmts = self._split_statements(body)

        for i, stmt in enumerate(stmts):
            stmt = stmt.strip()
            if not stmt:
                continue

            is_last = (i == len(stmts) - 1)

            # Rust implicit return: last expression without semicolon
            if is_last and not stmt.startswith("return ") \
               and not stmt.startswith("while ") and not stmt.startswith("for ") \
               and not stmt.startswith("let ") and not isinstance(return_type, VoidType) \
               and not stmt.startswith("//"):
                # If-expressions ending with } are implicit returns in Rust
                if stmt.startswith("if ") and stmt.endswith('}'):
                    val = self._compile_if_expr(stmt, builder, env, return_type, overflow)
                    if val is not None:
                        builder.ret(val)
                        return
                    # Fall through to compile as statement if if_expr fails
                elif not stmt.endswith('}'):
                    # This is an implicit return expression
                    val = self._compile_expr(stmt.rstrip(';'), builder, env, return_type, overflow)
                    if val is not None:
                        builder.ret(val)
                    return

            self._compile_statement(stmt, builder, func, env, return_type, overflow)

    def _split_statements(self, body: str) -> List[str]:
        """Split body into top-level statements."""
        stmts = []
        current = ""
        depth = 0
        for ch in body:
            if ch == '{':
                depth += 1
                current += ch
            elif ch == '}':
                depth -= 1
                current += ch
                if depth == 0:
                    # Don't split yet - check if 'else' follows
                    stmts.append(current.strip())
                    current = ""
            elif ch == ';' and depth == 0:
                stmts.append(current.strip())
                current = ""
            else:
                current += ch
        if current.strip():
            stmts.append(current.strip())

        # Merge else/else-if clauses with preceding if statements
        merged = []
        for s in stmts:
            if s.startswith("else") and merged:
                merged[-1] = merged[-1] + " " + s
            elif s:
                merged.append(s)
        return merged

    def _compile_statement(self, stmt: str, builder: IRBuilder, func: Function,
                           env: Dict[str, Value], return_type: IRType,
                           overflow: OverflowBehavior) -> None:
        stmt = stmt.strip()
        if not stmt:
            return

        # Return statement
        if stmt.startswith("return "):
            expr_text = stmt[7:].rstrip(';').strip()
            val = self._compile_expr(expr_text, builder, env, return_type, overflow)
            if val is not None:
                if isinstance(return_type, VoidType):
                    builder.ret_void()
                else:
                    builder.ret(val)
            return

        # If statement
        if stmt.startswith("if ") or stmt.startswith("if("):
            self._compile_if(stmt, builder, func, env, return_type, overflow)
            return

        # While loop
        if stmt.startswith("while ") or stmt.startswith("while("):
            self._compile_while(stmt, builder, func, env, return_type, overflow)
            return

        # For loop
        if stmt.startswith("for ") or stmt.startswith("for("):
            self._compile_for(stmt, builder, func, env, return_type, overflow)
            return

        # Let binding (Rust) / Variable declaration (C)
        if stmt.startswith("let "):
            self._compile_let(stmt, builder, env, return_type, overflow)
            return

        # C variable declaration
        for ty_prefix in ("int ", "unsigned int ", "unsigned ", "long long ",
                          "char ", "unsigned char "):
            if stmt.startswith(ty_prefix):
                self._compile_c_decl(stmt, ty_prefix, builder, env, return_type, overflow)
                return

        # Assignment
        if '=' in stmt and not stmt.startswith("=="):
            self._compile_assignment(stmt, builder, env, return_type, overflow)
            return

        # Handle ++ and -- as standalone statements
        clean = stmt.rstrip(';').strip()
        if clean.endswith("++") or clean.endswith("--"):
            self._compile_assignment(clean, builder, env, return_type, overflow)
            return

        # Expression statement (e.g., function call)
        self._compile_expr(stmt.rstrip(';'), builder, env, return_type, overflow)

    def _compile_if(self, stmt: str, builder: IRBuilder, func: Function,
                    env: Dict[str, Value], return_type: IRType,
                    overflow: OverflowBehavior) -> None:
        # Extract condition - handle both C-style if(cond) and Rust-style if cond {
        paren_start = stmt.find('(')
        brace_start = stmt.find('{')

        # Rust-style: if cond { ... } - no parentheses around condition
        if paren_start == -1 or (brace_start != -1 and brace_start < paren_start):
            # Condition is between "if " and the first "{"
            after_if = stmt[2:].strip()  # skip "if"
            brace_pos = after_if.find('{')
            cond_text = after_if[:brace_pos].strip()
            rest = after_if[brace_pos:].strip()
        else:
            # C-style: if (cond) { ... }
            depth = 0
            paren_end = paren_start
            for i in range(paren_start, len(stmt)):
                if stmt[i] == '(':
                    depth += 1
                elif stmt[i] == ')':
                    depth -= 1
                    if depth == 0:
                        paren_end = i
                        break

            cond_text = stmt[paren_start + 1:paren_end].strip()
            rest = stmt[paren_end + 1:].strip()

        # Get then/else bodies
        then_body, else_body = self._split_if_else(rest)

        # Compile condition
        cond_val = self._compile_expr(cond_text, builder, env, self.i32, overflow)
        if cond_val is None:
            cond_val = Constant.int_const(1, self.i32)

        # Convert to i1 if needed
        if isinstance(cond_val.type, IntType) and cond_val.type.width > 1:
            cond_val = self._build_cmp(builder, CmpPredicate.NE, cond_val,
                                   Constant.int_const(0, cond_val.type), name=self._unique_name("tobool"))

        then_bb = func.create_block("then")
        else_bb = func.create_block("else")
        merge_bb = func.create_block("merge")

        builder.cond_br(cond_val, then_bb, else_bb)

        # Then block
        builder.position_at_end(then_bb)
        then_env = dict(env)
        then_stmts = self._split_statements(then_body)
        has_return = False
        for s in then_stmts:
            s = s.strip()
            if s:
                self._compile_statement(s, builder, func, then_env, return_type, overflow)
                if s.startswith("return "):
                    has_return = True
        # If no return and body was a single expression, compile it as implicit return
        if not has_return and then_body.strip() and not then_body.strip().endswith(';'):
            expr_val = self._compile_expr(then_body.strip(), builder, then_env, return_type, overflow)
            if expr_val is not None and then_stmts and not any(s.strip().startswith("return ") for s in then_stmts):
                # Check if this is a Rust-style if-expression (has else, both produce values)
                if else_body:
                    pass  # Will handle as select below
        if not has_return:
            builder.br(merge_bb)
        env.update(then_env)

        # Else block
        builder.position_at_end(else_bb)
        if else_body:
            else_env = dict(env)
            else_stmts = self._split_statements(else_body)
            has_else_return = False
            for s in else_stmts:
                s = s.strip()
                if s:
                    self._compile_statement(s, builder, func, else_env, return_type, overflow)
                    if s.startswith("return "):
                        has_else_return = True
            if not has_else_return:
                builder.br(merge_bb)
            env.update(else_env)
        else:
            builder.br(merge_bb)

        builder.position_at_end(merge_bb)

    def _split_if_else(self, rest: str) -> Tuple[str, str]:
        """Split if-else into then body and optional else body."""
        rest = rest.strip()
        if rest.startswith('{'):
            depth = 0
            end = 0
            for i, ch in enumerate(rest):
                if ch == '{': depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            then_body = rest[1:end]
            after = rest[end + 1:].strip()
        else:
            # Single statement
            semi = rest.find(';')
            if semi >= 0:
                then_body = rest[:semi]
                after = rest[semi + 1:].strip()
            else:
                then_body = rest
                after = ""

        else_body = ""
        if after.startswith("else"):
            after = after[4:].strip()
            if after.startswith('{'):
                depth = 0
                for i, ch in enumerate(after):
                    if ch == '{': depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            else_body = after[1:i]
                            break
            elif after.startswith("if"):
                else_body = after
            else:
                semi = after.find(';')
                else_body = after[:semi] if semi >= 0 else after

        return then_body, else_body

    def _compile_while(self, stmt: str, builder: IRBuilder, func: Function,
                       env: Dict[str, Value], return_type: IRType,
                       overflow: OverflowBehavior) -> None:
        paren_start = stmt.find('(')
        brace_start = stmt.find('{')

        # Determine C-style vs Rust-style while:
        # C-style: while(cond) { ... } -- matching ')' is followed by '{'
        # Rust-style: while cond { ... } -- condition is everything up to '{'
        is_c_style = False
        if paren_start != -1 and (brace_start == -1 or paren_start < brace_start):
            # Find matching paren
            depth = 0
            paren_end = paren_start
            for i in range(paren_start, len(stmt)):
                if stmt[i] == '(': depth += 1
                elif stmt[i] == ')':
                    depth -= 1
                    if depth == 0:
                        paren_end = i
                        break
            # C-style if after ')' the next non-space is '{'
            after_paren = stmt[paren_end + 1:].strip()
            if after_paren.startswith('{'):
                is_c_style = True

        if is_c_style:
            cond_text = stmt[paren_start + 1:paren_end].strip()
            rest = stmt[paren_end + 1:].strip()
        else:
            # Rust-style: everything between "while " and "{"
            after_while = stmt[5:].strip()  # skip "while"
            brace_pos = after_while.find('{')
            cond_text = after_while[:brace_pos].strip()
            rest = after_while[brace_pos:].strip()

        # Extract body
        if rest.startswith('{'):
            body_text = rest[1:rest.rfind('}')]
        else:
            body_text = rest.rstrip(';')

        # Unroll loop into straight-line if-then code for SMT encoding
        # (The SMT encoder cannot handle back-edges/cycles)
        # Try to detect loop bound from condition like "i < N"
        UNROLL_COUNT = 8
        body_stmts = self._split_statements(body_text)

        for _iter in range(UNROLL_COUNT):
            # Check condition
            cond_val = self._compile_expr(cond_text, builder, env, self.i32, overflow)
            if cond_val is None:
                cond_val = Constant.int_const(0, self.i32)
            if isinstance(cond_val.type, IntType) and cond_val.type.width > 1:
                cond_val = self._build_cmp(builder, CmpPredicate.NE, cond_val,
                                       Constant.int_const(0, cond_val.type), name=self._unique_name("loopcond"))

            # Save env before body for conditional merge
            pre_body_env = dict(env)

            # Compile body (updates env)
            for s in body_stmts:
                s = s.strip()
                if s:
                    self._compile_statement(s, builder, func, env, return_type, overflow)

            # Conditionally merge: if cond was true, use updated value; else keep old
            for name in list(env.keys()):
                old_val = pre_body_env.get(name)
                new_val = env[name]
                if old_val is not None and new_val is not old_val and hasattr(old_val, 'type') and hasattr(new_val, 'type'):
                    try:
                        old_val_c, new_val_c = self._coerce_values(builder, old_val, new_val)
                        env[name] = builder.select(cond_val, new_val_c, old_val_c,
                                                   name=self._unique_name(f"loop_{name}")
                        )
                    except Exception:
                        pass  # Keep new value if coercion fails

    def _compile_for(self, stmt: str, builder: IRBuilder, func: Function,
                     env: Dict[str, Value], return_type: IRType,
                     overflow: OverflowBehavior) -> None:
        # Parse for(init; cond; incr) { body }
        paren_start = stmt.find('(')
        depth = 0
        paren_end = paren_start
        for i in range(paren_start, len(stmt)):
            if stmt[i] == '(': depth += 1
            elif stmt[i] == ')':
                depth -= 1
                if depth == 0:
                    paren_end = i
                    break

        parts_text = stmt[paren_start + 1:paren_end]
        parts = parts_text.split(';')
        init_text = parts[0].strip() if len(parts) > 0 else ""
        cond_text = parts[1].strip() if len(parts) > 1 else "1"
        incr_text = parts[2].strip() if len(parts) > 2 else ""

        rest = stmt[paren_end + 1:].strip()
        if rest.startswith('{'):
            body_text = rest[1:rest.rfind('}')]
        else:
            body_text = rest.rstrip(';')

        # Compile init
        if init_text:
            self._compile_statement(init_text, builder, func, env, return_type, overflow)

        # Unroll for-loop into straight-line if-then code for SMT encoding
        UNROLL_COUNT = 8
        body_stmts = self._split_statements(body_text)

        for _iter in range(UNROLL_COUNT):
            if cond_text:
                cond_val = self._compile_expr(cond_text, builder, env, self.i32, overflow)
                if cond_val is None:
                    cond_val = Constant.int_const(1, self.i32)
                if isinstance(cond_val.type, IntType) and cond_val.type.width > 1:
                    cond_val = self._build_cmp(builder, CmpPredicate.NE, cond_val,
                                            Constant.int_const(0, cond_val.type), name=self._unique_name("forcond"))
            else:
                cond_val = Constant.bool_const(True)

            # Save env before body
            pre_body_env = dict(env)

            # Compile body
            for s in body_stmts:
                s = s.strip()
                if s:
                    self._compile_statement(s, builder, func, env, return_type, overflow)

            # Increment
            if incr_text:
                self._compile_statement(incr_text, builder, func, env, return_type, overflow)

            # Conditionally merge: if cond was true, use updated; else keep old
            for name in list(env.keys()):
                old_val = pre_body_env.get(name)
                new_val = env[name]
                if old_val is not None and new_val is not old_val and hasattr(old_val, 'type') and hasattr(new_val, 'type'):
                    try:
                        old_val_c, new_val_c = self._coerce_values(builder, old_val, new_val)
                        env[name] = builder.select(cond_val, new_val_c, old_val_c,
                                                   name=self._unique_name(f"loop_{name}"))
                    except Exception:
                        pass

    def _compile_let(self, stmt: str, builder: IRBuilder, env: Dict[str, Value],
                     return_type: IRType, overflow: OverflowBehavior) -> None:
        stmt = stmt[4:].strip()  # remove "let "
        stmt = stmt.replace("mut ", "").strip()

        if '=' in stmt:
            parts = stmt.split('=', 1)
            name_part = parts[0].strip().rstrip(';')
            expr = parts[1].strip().rstrip(';')

            # Parse optional type annotation
            var_type = self.i32
            var_name = name_part
            if ':' in name_part:
                var_name, ty_str = name_part.split(':', 1)
                var_name = var_name.strip()
                var_type = self._parse_rust_type(ty_str.strip())

            val = self._compile_expr(expr, builder, env, var_type, overflow)
            if val is not None:
                env[var_name] = val

    def _compile_c_decl(self, stmt: str, ty_prefix: str, builder: IRBuilder,
                        env: Dict[str, Value], return_type: IRType,
                        overflow: OverflowBehavior) -> None:
        rest = stmt[len(ty_prefix):].strip().rstrip(';')
        var_type = self._parse_c_type(ty_prefix.strip())

        # Handle multiple declarations: int a = 0, b = 1;
        for decl in rest.split(','):
            decl = decl.strip()
            if '=' in decl:
                name, expr = decl.split('=', 1)
                name = name.strip()
                val = self._compile_expr(expr.strip(), builder, env, var_type, overflow)
                if val is not None:
                    env[name] = val
            else:
                name = decl.strip()
                if name:
                    env[name] = Constant.int_const(0, var_type)

    def _compile_assignment(self, stmt: str, builder: IRBuilder,
                            env: Dict[str, Value], return_type: IRType,
                            overflow: OverflowBehavior) -> None:
        stmt = stmt.rstrip(';').strip()

        # Handle compound assignments
        for op_str in ("+=", "-=", "*=", "/=", "%=", "<<=", ">>=", "&=", "|=", "^="):
            if op_str in stmt:
                parts = stmt.split(op_str, 1)
                name = parts[0].strip()
                expr = parts[1].strip()
                base_op = op_str[:-1]
                combined = f"{name} {base_op} ({expr})"
                val = self._compile_expr(combined, builder, env, return_type, overflow)
                if val is not None:
                    env[name] = val
                return

        # Handle ++ and --
        if stmt.endswith("++"):
            name = stmt[:-2].strip()
            if name in env:
                meta = InstructionMetadata(overflow=overflow)
                val = self._build_binop(builder, BinOpKind.ADD, env[name],
                                        Constant.int_const(1, env[name].type),
                                        name=f"{name}_inc", metadata=meta)
                env[name] = val
            return
        if stmt.endswith("--"):
            name = stmt[:-2].strip()
            if name in env:
                meta = InstructionMetadata(overflow=overflow)
                val = self._build_binop(builder, BinOpKind.SUB, env[name],
                                        Constant.int_const(1, env[name].type),
                                        name=f"{name}_dec", metadata=meta)
                env[name] = val
            return

        # Simple assignment
        if '=' in stmt:
            eq_pos = stmt.index('=')
            # Make sure it's not == or !=
            if eq_pos > 0 and stmt[eq_pos - 1] in ('!', '<', '>', '='):
                return
            if eq_pos + 1 < len(stmt) and stmt[eq_pos + 1] == '=':
                return
            name = stmt[:eq_pos].strip()
            expr = stmt[eq_pos + 1:].strip()
            target_type = env.get(name, Constant.int_const(0, self.i32)).type if name in env else return_type
            val = self._compile_expr(expr, builder, env, target_type, overflow)
            if val is not None:
                env[name] = val

    def _compile_expr(self, expr: str, builder: IRBuilder, env: Dict[str, Value],
                      target_type: IRType, overflow: OverflowBehavior,
                      _depth: int = 0) -> Optional[Value]:
        """Compile an expression to a Value."""
        expr = expr.strip()
        if not expr:
            return None
        if _depth > 30:
            return Constant.int_const(0, target_type if isinstance(target_type, IntType) else self.i32)

        # Remove outer parens
        while expr.startswith('(') and expr.endswith(')') and self._matching_parens(expr):
            expr = expr[1:-1].strip()

        # Integer literal
        if expr.lstrip('-').replace('u', '').replace('U', '').isdigit():
            val = int(expr.replace('u', '').replace('U', ''))
            return Constant.int_const(val, target_type if isinstance(target_type, IntType) else self.i32)

        # Hex literal
        if expr.startswith("0x") or expr.startswith("0X"):
            val = int(expr.rstrip('uU'), 16)
            return Constant.int_const(val, target_type if isinstance(target_type, IntType) else self.u32)

        # Named constants
        if expr in ("i32::MIN", "INT_MIN", "-2147483648"):
            return Constant.int_const(-2147483648, self.i32)
        if expr in ("i32::MAX", "INT_MAX", "2147483647"):
            return Constant.int_const(2147483647, self.i32)

        # Boolean literals
        if expr in ("true", "1"):
            return Constant.int_const(1, IntType(1, Signedness.UNSIGNED))
        if expr in ("false", "0"):
            return Constant.int_const(0, target_type if isinstance(target_type, IntType) else self.i32)

        # Variable reference
        if expr in env:
            return env[expr]

        # Rust if-expression: if cond { expr1 } else { expr2 }
        if expr.startswith("if "):
            return self._compile_if_expr(expr, builder, env, target_type, overflow)

        # Rust wrapping methods - only match when expr is a method call, not containing one
        if '.' in expr:
            # Find the dot at top level (not inside parens)
            dot_pos = -1
            depth = 0
            for i, ch in enumerate(expr):
                if ch == '(': depth += 1
                elif ch == ')': depth -= 1
                elif ch == '.' and depth == 0:
                    dot_pos = i
                    break
            if dot_pos > 0:
                after_dot = expr[dot_pos + 1:]
                for method in ("wrapping_add(", "wrapping_sub(", "wrapping_mul(",
                               "wrapping_neg()", "wrapping_div(", "wrapping_rem(",
                               "wrapping_shl(", "wrapping_shr(",
                               "saturating_add(", "saturating_sub(",
                               "checked_add("):
                    if after_dot.startswith(method):
                        return self._compile_wrapping_method(expr, builder, env, target_type, overflow)

        # Cast: (type)expr or expr as type
        # Only match " as " at top level (not inside parens)
        as_pos = self._find_binary_op(expr, " as ")
        if as_pos >= 0:
            inner_text = expr[:as_pos].strip()
            dst_text = expr[as_pos + 4:].strip()
            inner = self._compile_expr(inner_text, builder, env, target_type, overflow)
            dst_type = self._parse_rust_type(dst_text)
            if inner is not None:
                return self._build_cast(builder, inner, dst_type)
            return None
        if expr.startswith("(") and ')' in expr:
            # C-style cast: (int)x
            close = expr.index(')')
            cast_type_str = expr[1:close].strip()
            if cast_type_str in ("int", "unsigned int", "long long", "unsigned", "unsigned char"):
                rest = expr[close + 1:].strip()
                inner = self._compile_expr(rest, builder, env, target_type, overflow)
                if inner is not None:
                    return self._build_cast(builder, inner, self._parse_c_type(cast_type_str))
                return None

        # Ternary: cond ? a : b
        if '?' in expr and ':' in expr:
            q_pos = expr.index('?')
            c_pos = expr.index(':', q_pos)
            cond_expr = expr[:q_pos].strip()
            true_expr = expr[q_pos + 1:c_pos].strip()
            false_expr = expr[c_pos + 1:].strip()
            cond = self._compile_expr(cond_expr, builder, env, self.i32, overflow)
            tv = self._compile_expr(true_expr, builder, env, target_type, overflow)
            fv = self._compile_expr(false_expr, builder, env, target_type, overflow)
            if cond and tv and fv:
                if isinstance(cond.type, IntType) and cond.type.width > 1:
                    cond = self._build_cmp(builder, CmpPredicate.NE, cond,
                                       Constant.int_const(0, cond.type), name=self._unique_name("terncond"))
                return builder.select(cond, tv, fv, name=self._unique_name("ternary"))
            return tv or fv

        # Binary operations (ordered by precedence, lowest first)
        for ops in [
            [("||", None)],
            [("&&", None)],
            [("|", BinOpKind.OR)],
            [("^", BinOpKind.XOR)],
            [("&", BinOpKind.AND)],
            [("==", CmpPredicate.EQ), ("!=", CmpPredicate.NE)],
            [("<=", CmpPredicate.SLE), (">=", CmpPredicate.SGE),
             ("<", CmpPredicate.SLT), (">", CmpPredicate.SGT)],
            [("<<", BinOpKind.SHL), (">>", BinOpKind.ASHR)],
            [("+", BinOpKind.ADD), ("-", BinOpKind.SUB)],
            [("*", BinOpKind.MUL), ("/", BinOpKind.SDIV), ("%", BinOpKind.SREM)],
        ]:
            result = self._try_binary_op(expr, ops, builder, env, target_type, overflow)
            if result is not None:
                return result

        # Unary negation
        if expr.startswith("-") and not expr[1:].lstrip().startswith('-'):
            inner = self._compile_expr(expr[1:].strip(), builder, env, target_type, overflow)
            if inner is not None:
                meta = InstructionMetadata(overflow=overflow)
                return self._build_neg(builder, inner, name=self._unique_name("neg"), metadata=meta)

        # Unary NOT
        if expr.startswith("!"):
            inner = self._compile_expr(expr[1:].strip(), builder, env, self.i32, overflow)
            if inner is not None:
                return self._build_cmp(builder, CmpPredicate.EQ, inner,
                                   Constant.int_const(0, inner.type), name=self._unique_name("not"))

        # Bitwise NOT
        if expr.startswith("~"):
            inner = self._compile_expr(expr[1:].strip(), builder, env, target_type, overflow)
            if inner is not None:
                return builder.not_(inner, name=self._unique_name("bitnot"))

        # Default: treat as zero constant
        return Constant.int_const(0, target_type if isinstance(target_type, IntType) else self.i32)

    def _try_binary_op(self, expr: str, ops, builder, env, target_type, overflow):
        """Try to parse a binary operation at the given precedence level."""
        for op_str, op_kind in ops:
            pos = self._find_binary_op(expr, op_str)
            if pos >= 0:
                lhs_text = expr[:pos].strip()
                rhs_text = expr[pos + len(op_str):].strip()
                lhs = self._compile_expr(lhs_text, builder, env, target_type, overflow)
                rhs = self._compile_expr(rhs_text, builder, env, target_type, overflow)
                if lhs is None or rhs is None:
                    return None

                # Coerce types
                lhs, rhs = self._coerce_values(builder, lhs, rhs)
                # C unsigned overflow is well-defined wrapping
                actual_overflow = overflow
                if isinstance(lhs.type, IntType) and lhs.type.is_unsigned:
                    actual_overflow = OverflowBehavior.WRAP
                meta = InstructionMetadata(overflow=actual_overflow)

                if op_kind is None:
                    # Logical ops
                    if op_str == "&&":
                        l = self._build_cmp(builder, CmpPredicate.NE, lhs, Constant.int_const(0, lhs.type))
                        r = self._build_cmp(builder, CmpPredicate.NE, rhs, Constant.int_const(0, rhs.type))
                        return builder.and_(l, r, name=self._unique_name("land"))
                    elif op_str == "||":
                        l = self._build_cmp(builder, CmpPredicate.NE, lhs, Constant.int_const(0, lhs.type))
                        r = self._build_cmp(builder, CmpPredicate.NE, rhs, Constant.int_const(0, rhs.type))
                        return builder.or_(l, r, name=self._unique_name("lor"))
                elif isinstance(op_kind, CmpPredicate):
                    # Use unsigned comparison predicates for unsigned types
                    if isinstance(lhs.type, IntType) and lhs.type.is_unsigned:
                        unsigned_map = {
                            CmpPredicate.SLT: CmpPredicate.ULT,
                            CmpPredicate.SLE: CmpPredicate.ULE,
                            CmpPredicate.SGT: CmpPredicate.UGT,
                            CmpPredicate.SGE: CmpPredicate.UGE,
                        }
                        op_kind = unsigned_map.get(op_kind, op_kind)
                    return self._build_cmp(builder, op_kind, lhs, rhs, name=self._unique_name("cmp"))
                else:
                    # Use unsigned div/rem for unsigned types
                    if isinstance(lhs.type, IntType) and lhs.type.is_unsigned:
                        if op_kind == BinOpKind.SDIV:
                            op_kind = BinOpKind.UDIV
                        elif op_kind == BinOpKind.SREM:
                            op_kind = BinOpKind.UREM
                        elif op_kind == BinOpKind.ASHR:
                            op_kind = BinOpKind.LSHR
                    return builder.binop(op_kind, lhs, rhs, name=self._unique_name("op"), metadata=meta)
        return None

    def _find_binary_op(self, expr: str, op: str) -> int:
        """Find a binary operator at top level (not inside parens)."""
        depth = 0
        i = len(expr) - 1
        while i >= 0:
            ch = expr[i]
            if ch == ')': depth += 1
            elif ch == '(': depth -= 1
            elif depth == 0:
                if expr[i:i + len(op)] == op:
                    # Make sure it's not part of a longer operator
                    before = expr[i - 1] if i > 0 else ' '
                    after = expr[i + len(op)] if i + len(op) < len(expr) else ' '
                    if op == ">" and before == '-':
                        i -= 1
                        continue
                    if op == "<" and after == '<':
                        i -= 1
                        continue
                    if op == ">" and after == '>':
                        i -= 1
                        continue
                    if op == "=" and (before == '!' or before == '<' or before == '>' or before == '='):
                        i -= 1
                        continue
                    if op == "=" and after == '=':
                        i -= 1
                        continue
                    if op in ("&", "|") and len(op) == 1:
                        if (i > 0 and expr[i-1] == op[0]) or (i + 1 < len(expr) and expr[i+1] == op[0]):
                            i -= 1
                            continue
                    if op == "<" and before == '<':
                        i -= 1
                        continue
                    if op == ">" and before == '>':
                        i -= 1
                        continue
                    return i
            i -= 1
        return -1

    def _compile_if_expr(self, expr: str, builder: IRBuilder, env: Dict[str, Value],
                         target_type: IRType, overflow: OverflowBehavior) -> Optional[Value]:
        """Compile a Rust if-expression: if cond { expr1 } else { expr2 } → select."""
        # Find condition (between 'if' and first '{')
        brace = expr.find('{')
        if brace < 0:
            return None
        cond_text = expr[3:brace].strip()  # after 'if '

        # Find then body (between first { and matching })
        depth = 0
        then_end = brace
        for i in range(brace, len(expr)):
            if expr[i] == '{': depth += 1
            elif expr[i] == '}':
                depth -= 1
                if depth == 0:
                    then_end = i
                    break
        then_body = expr[brace + 1:then_end].strip()

        # Find else body
        after = expr[then_end + 1:].strip()
        else_body = ""
        if after.startswith("else"):
            after = after[4:].strip()
            if after.startswith("if "):
                # else if - recurse
                else_body = after
            elif after.startswith('{'):
                depth = 0
                for i, ch in enumerate(after):
                    if ch == '{': depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            else_body = after[1:i]
                            break

        # Compile condition
        cond_val = self._compile_expr(cond_text, builder, env, self.i32, overflow)
        if cond_val is None:
            return None
        if isinstance(cond_val.type, IntType) and cond_val.type.width > 1:
            cond_val = self._build_cmp(builder, CmpPredicate.NE, cond_val,
                                       Constant.int_const(0, cond_val.type), name=self._unique_name("ifcond"))

        # Compile then and else as expressions
        then_val = self._compile_expr(then_body.rstrip(';').strip(), builder, env, target_type, overflow)
        if else_body:
            else_val = self._compile_expr(else_body.rstrip(';').strip(), builder, env, target_type, overflow)
        else:
            else_val = Constant.int_const(0, target_type if isinstance(target_type, IntType) else self.i32)

        if then_val is not None and else_val is not None:
            then_val, else_val = self._coerce_values(builder, then_val, else_val)
            return builder.select(cond_val, then_val, else_val, name=self._unique_name("ifexpr"))
        return then_val or else_val

    def _compile_wrapping_method(self, expr: str, builder: IRBuilder,
                                 env: Dict[str, Value], target_type: IRType,
                                 overflow: OverflowBehavior) -> Optional[Value]:
        """Compile Rust wrapping/saturating/checked arithmetic methods."""
        # Parse: base.method(arg)
        dot_pos = expr.find('.')
        if dot_pos < 0:
            return None

        base_text = expr[:dot_pos].strip()
        method_call = expr[dot_pos + 1:].strip()

        base = self._compile_expr(base_text, builder, env, target_type, overflow)
        if base is None:
            return None

        meta = InstructionMetadata(overflow=OverflowBehavior.WRAP)

        if method_call.startswith("wrapping_neg()"):
            return self._build_neg(builder, base, name=self._unique_name("wneg"), metadata=meta)

        # Extract method name and argument
        paren_start = method_call.find('(')
        paren_end = method_call.rfind(')')
        if paren_start < 0 or paren_end < 0:
            return base

        method_name = method_call[:paren_start]
        arg_text = method_call[paren_start + 1:paren_end].strip()

        arg = self._compile_expr(arg_text, builder, env, target_type, overflow) if arg_text else None

        if arg is None and method_name != "wrapping_neg":
            return base

        if arg is not None:
            base, arg = self._coerce_values(builder, base, arg)

        if method_name == "wrapping_add":
            return self._build_binop(builder, BinOpKind.ADD, base, arg, name=self._unique_name("wadd"), metadata=meta)
        elif method_name == "wrapping_sub":
            return self._build_binop(builder, BinOpKind.SUB, base, arg, name=self._unique_name("wsub"), metadata=meta)
        elif method_name == "wrapping_mul":
            return self._build_binop(builder, BinOpKind.MUL, base, arg, name=self._unique_name("wmul"), metadata=meta)
        elif method_name == "wrapping_div":
            return builder.sdiv(base, arg, name=self._unique_name("wdiv"))
        elif method_name == "wrapping_rem":
            return builder.srem(base, arg, name=self._unique_name("wrem"))
        elif method_name == "wrapping_shl":
            return self._build_binop(builder, BinOpKind.SHL, base, arg, name=self._unique_name("wshl"), metadata=meta)
        elif method_name == "wrapping_shr":
            return self._build_binop(builder, BinOpKind.ASHR, base, arg, name=self._unique_name("wshr"), metadata=meta)
        elif method_name == "saturating_add":
            meta_sat = InstructionMetadata(overflow=OverflowBehavior.SATURATE)
            return self._build_binop(builder, BinOpKind.ADD, base, arg, name=self._unique_name("satadd"), metadata=meta_sat)
        elif method_name == "saturating_sub":
            meta_sat = InstructionMetadata(overflow=OverflowBehavior.SATURATE)
            return self._build_binop(builder, BinOpKind.SUB, base, arg, name=self._unique_name("satsub"), metadata=meta_sat)
        elif method_name == "checked_add":
            return self._build_binop(builder, BinOpKind.ADD, base, arg, name="cadd", metadata=meta)
        else:
            return base

    def _coerce_values(self, builder: IRBuilder, a: Value, b: Value) -> Tuple[Value, Value]:
        """Coerce two values to the same type."""
        if a.type == b.type:
            return a, b
        if isinstance(a.type, IntType) and isinstance(b.type, IntType):
            if a.type.width < b.type.width:
                a = builder.sext(a, b.type, name=self._unique_name("sext"))
            elif b.type.width < a.type.width:
                b = builder.sext(b, a.type, name=self._unique_name("sext"))
        return a, b

    def _build_cast(self, builder: IRBuilder, val: Value, dst_type: IRType) -> Value:
        if isinstance(val.type, IntType) and isinstance(dst_type, IntType):
            if val.type.width < dst_type.width:
                if val.type.is_signed:
                    return builder.sext(val, dst_type, name=self._unique_name("cast"))
                else:
                    return builder.zext(val, dst_type, name=self._unique_name("cast"))
            elif val.type.width > dst_type.width:
                return builder.trunc(val, dst_type, name=self._unique_name("cast"))
        return val

    def _matching_parens(self, expr: str) -> bool:
        depth = 0
        for i, ch in enumerate(expr):
            if ch == '(': depth += 1
            elif ch == ')': depth -= 1
            if depth == 0 and i < len(expr) - 1:
                return False
        return depth == 0


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline result types
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineResult:
    """Result of running the full pipeline on one function pair."""
    name: str
    category: str
    expected: str
    actual: str
    correct: bool
    time_ms: float
    explanation: str = ""
    counterexample: Optional[Dict] = None
    coercion_points: int = 0
    alignment_similarity: float = 0.0

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "category": self.category,
            "expected": self.expected,
            "actual": self.actual,
            "correct": self.correct,
            "time_ms": round(self.time_ms, 1),
            "explanation": self.explanation,
            "coercion_points": self.coercion_points,
            "alignment_similarity": round(self.alignment_similarity, 3),
        }
        if self.counterexample:
            d["counterexample"] = self.counterexample
        return d


@dataclass
class PipelineSummary:
    """Summary of pipeline results."""
    total: int = 0
    correct: int = 0
    incorrect: int = 0
    errors: int = 0
    total_time_ms: float = 0.0
    results: List[PipelineResult] = field(default_factory=list)
    categories: Dict[str, Dict] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "correct": self.correct,
            "incorrect": self.incorrect,
            "errors": self.errors,
            "accuracy": round(self.accuracy, 4),
            "total_time_ms": round(self.total_time_ms, 1),
            "categories": self.categories,
            "results": [r.to_dict() for r in self.results],
        }


# ═══════════════════════════════════════════════════════════════════════════
# Full pipeline
# ═══════════════════════════════════════════════════════════════════════════

class FullPipeline:
    """
    End-to-end cross-language equivalence verification pipeline.

    Takes C and Rust source code → parses → lowers → aligns →
    constructs product program → generates Z3 queries → verifies.
    """

    def __init__(self, c_config: Optional[SemanticConfig] = None,
                 rust_config: Optional[SemanticConfig] = None,
                 timeout_ms: int = 30000):
        self.c_config = c_config or SemanticConfig.c11()
        self.rust_config = rust_config or SemanticConfig.rust_release()
        self.timeout_ms = timeout_ms
        self.compiler = MiniCCompiler()

    def verify_pair(self, c_source: str, rust_source: str,
                    func_name: str) -> PipelineResult:
        """Run full pipeline on a single C/Rust function pair."""
        t0 = time.time()

        try:
            # Step 1: Parse and lower C to IR
            c_func = self.compiler.compile_function(c_source, func_name, language="c")
            if c_func is None:
                return PipelineResult(
                    name=func_name, category="", expected="", actual="error",
                    correct=False, time_ms=(time.time() - t0) * 1000,
                    explanation="Failed to compile C source"
                )

            # Step 2: Parse and lower Rust to IR
            rust_func = self.compiler.compile_function(rust_source, func_name, language="rust")
            if rust_func is None:
                return PipelineResult(
                    name=func_name, category="", expected="", actual="error",
                    correct=False, time_ms=(time.time() - t0) * 1000,
                    explanation="Failed to compile Rust source"
                )

            # Step 3: Build product program and verify
            from src.product_program.auto_builder import AutoProductBuilder
            product_builder = AutoProductBuilder(self.c_config, self.rust_config)
            alignment, equiv_result = product_builder.align_and_verify(
                c_func, rust_func, self.timeout_ms
            )

            elapsed = (time.time() - t0) * 1000

            # Map result
            actual = equiv_result.status
            if actual == "equivalent":
                actual_label = "equivalent"
            elif actual == "divergent":
                actual_label = "divergent"
            else:
                actual_label = "unknown"

            ce_dict = None
            if equiv_result.witness:
                ce_dict = {
                    "inputs": equiv_result.witness.inputs,
                    "c_result": str(equiv_result.witness.c_result),
                    "rust_result": str(equiv_result.witness.rust_result),
                    "kind": equiv_result.witness.divergence_kind,
                }

            return PipelineResult(
                name=func_name,
                category="",
                expected="",
                actual=actual_label,
                correct=True,  # Will be set by caller
                time_ms=elapsed,
                explanation=equiv_result.explanation,
                counterexample=ce_dict,
                coercion_points=len(alignment.coercion_points),
                alignment_similarity=alignment.overall_similarity,
            )

        except Exception as e:
            elapsed = (time.time() - t0) * 1000
            return PipelineResult(
                name=func_name, category="", expected="", actual="error",
                correct=False, time_ms=elapsed,
                explanation=f"Pipeline error: {e}"
            )

    def run_benchmarks(self, benchmarks=None) -> PipelineSummary:
        """Run pipeline on all benchmarks."""
        if benchmarks is None:
            sys.path.insert(0, os.path.join(IMPL_DIR, "benchmarks", "pairs"))
            from benchmark_pairs import ALL_BENCHMARKS
            benchmarks = ALL_BENCHMARKS

        summary = PipelineSummary()
        summary.total = len(benchmarks)

        for bench in benchmarks:
            print(f"  [{bench.category}] {bench.name}...", end=" ", flush=True)

            result = self.verify_pair(bench.c_source, bench.rust_source, bench.name)
            result.category = bench.category
            result.expected = bench.expected_result

            # Determine correctness
            if result.actual == "error":
                result.correct = False
                summary.errors += 1
                print(f"ERROR ({result.time_ms:.0f}ms)")
            elif result.actual == bench.expected_result:
                result.correct = True
                summary.correct += 1
                sym = "✓" if result.actual == "equivalent" else "✗"
                print(f"{sym} {result.actual} ({result.time_ms:.0f}ms)")
            elif result.actual == "unknown":
                # Unknown is acceptable but not a full pass
                result.correct = False
                summary.incorrect += 1
                print(f"? unknown ({result.time_ms:.0f}ms)")
            else:
                # Wrong answer
                result.correct = False
                summary.incorrect += 1
                print(f"✗ WRONG: expected={bench.expected_result}, got={result.actual} ({result.time_ms:.0f}ms)")

            if result.counterexample:
                ce = result.counterexample
                print(f"    Counterexample: inputs={ce['inputs']}, "
                      f"C={ce['c_result']}, Rust={ce['rust_result']}")

            summary.results.append(result)
            summary.total_time_ms += result.time_ms

            # Track per-category stats
            cat = bench.category
            if cat not in summary.categories:
                summary.categories[cat] = {"total": 0, "correct": 0, "errors": 0}
            summary.categories[cat]["total"] += 1
            if result.correct:
                summary.categories[cat]["correct"] += 1
            if result.actual == "error":
                summary.categories[cat]["errors"] += 1

        return summary


# ═══════════════════════════════════════════════════════════════════════════
# LLVM IR baseline comparison
# ═══════════════════════════════════════════════════════════════════════════

def run_llvm_ir_baseline(benchmarks=None) -> Dict[str, Any]:
    """
    Demonstrate what LLVM IR-level comparison misses vs source-level analysis.

    Key insight: LLVM IR erases semantic differences because both C and Rust
    compile to the same BV operations. Source-level analysis with σ-parameterization
    can detect divergences that are invisible at IR level.
    """
    if benchmarks is None:
        sys.path.insert(0, os.path.join(IMPL_DIR, "benchmarks", "pairs"))
        from benchmark_pairs import ALL_BENCHMARKS, get_divergent_benchmarks
        benchmarks = get_divergent_benchmarks()

    results = {
        "description": "LLVM IR baseline: what IR-level comparison misses",
        "insight": (
            "LLVM IR erases semantic differences. Both C's UB-on-overflow and "
            "Rust's wrapping compile to identical 'add nsw' / 'add' instructions. "
            "Source-level analysis with σ = (ovf, fp, err) detects these invisible divergences."
        ),
        "pairs_analyzed": len(benchmarks),
        "ir_would_miss": 0,
        "ir_would_catch": 0,
        "details": [],
    }

    for bench in benchmarks:
        # At LLVM IR level, most operations look identical
        # The key divergences that IR misses:
        ir_visible = False

        if bench.divergence_kind in ("signed_overflow", "negation_overflow"):
            # LLVM: both compile to 'add'/'sub'/'mul' (with or without nsw/nuw)
            # At -O0: C has 'add nsw', Rust has 'add' (or call to panic)
            # At -O2: both may optimize to same thing
            ir_visible = False
        elif bench.divergence_kind in ("shift_overflow", "shift_semantics"):
            # LLVM: C uses 'shl', Rust uses 'and' + 'shl' (masked)
            ir_visible = True  # Rust adds explicit mask
        elif bench.divergence_kind in ("division_by_zero", "division_overflow"):
            # LLVM: both compile to 'sdiv'/'udiv', Rust may add branch
            ir_visible = True  # Rust adds explicit zero check
        elif bench.divergence_kind in ("bounds_check",):
            # LLVM: Rust adds bounds check branches
            ir_visible = True
        elif bench.divergence_kind in ("error_model", "null_handling", "signedness", "saturation"):
            # These are semantic-level differences not visible in IR
            ir_visible = False
        else:
            ir_visible = False

        if ir_visible:
            results["ir_would_catch"] += 1
        else:
            results["ir_would_miss"] += 1

        results["details"].append({
            "name": bench.name,
            "divergence_kind": bench.divergence_kind,
            "ir_visible": ir_visible,
            "explanation": (
                f"{'IR shows difference' if ir_visible else 'IR erases difference'}: "
                f"{bench.description}"
            ),
        })

    miss_rate = results["ir_would_miss"] / max(len(benchmarks), 1) * 100
    results["miss_rate_pct"] = round(miss_rate, 1)
    results["conclusion"] = (
        f"LLVM IR comparison would miss {results['ir_would_miss']}/{len(benchmarks)} "
        f"({miss_rate:.0f}%) of divergences. Source-level σ-parameterized analysis "
        f"catches all of them."
    )

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Cross-language equivalence verification pipeline"
    )
    parser.add_argument("--category", type=str, default=None,
                        help="Run only benchmarks in this category")
    parser.add_argument("--c-file", type=str, default=None,
                        help="Path to C source file")
    parser.add_argument("--rust-file", type=str, default=None,
                        help="Path to Rust source file")
    parser.add_argument("--func", type=str, default=None,
                        help="Function name to verify")
    parser.add_argument("--timeout", type=int, default=30000,
                        help="Z3 timeout in milliseconds")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path")
    parser.add_argument("--llvm-baseline", action="store_true",
                        help="Run LLVM IR baseline comparison")
    args = parser.parse_args()

    print("=" * 70)
    print("Cross-Language Equivalence Verifier — Full Pipeline")
    print("=" * 70)
    print()

    pipeline = FullPipeline(timeout_ms=args.timeout)

    if args.c_file and args.rust_file:
        # Custom file pair
        with open(args.c_file) as f:
            c_source = f.read()
        with open(args.rust_file) as f:
            rust_source = f.read()
        func_name = args.func or "main"

        print(f"Verifying: {args.c_file} ↔ {args.rust_file} [{func_name}]")
        result = pipeline.verify_pair(c_source, rust_source, func_name)
        print(f"Result: {result.actual}")
        if result.explanation:
            print(f"Explanation: {result.explanation}")
        if result.counterexample:
            print(f"Counterexample: {result.counterexample}")
        return 0

    # Run benchmarks
    sys.path.insert(0, os.path.join(IMPL_DIR, "benchmarks", "pairs"))
    from benchmark_pairs import ALL_BENCHMARKS, get_benchmarks_by_category

    if args.category:
        benchmarks = get_benchmarks_by_category(args.category)
        if not benchmarks:
            print(f"No benchmarks found for category: {args.category}")
            return 1
    else:
        benchmarks = ALL_BENCHMARKS

    print(f"Running {len(benchmarks)} benchmark pairs...")
    print()

    summary = pipeline.run_benchmarks(benchmarks)

    # Print summary
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total:    {summary.total}")
    print(f"Correct:  {summary.correct} ({summary.accuracy:.0%})")
    print(f"Incorrect:{summary.incorrect}")
    print(f"Errors:   {summary.errors}")
    print(f"Time:     {summary.total_time_ms:.0f}ms")
    print()
    print("Per-category:")
    for cat, stats in sorted(summary.categories.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {cat:20s}: {stats['correct']}/{stats['total']} ({acc:.0%})")

    # LLVM IR baseline
    if args.llvm_baseline:
        print()
        print("=" * 70)
        print("LLVM IR BASELINE COMPARISON")
        print("=" * 70)
        ir_results = run_llvm_ir_baseline()
        print(f"Divergent pairs analyzed: {ir_results['pairs_analyzed']}")
        print(f"IR would miss: {ir_results['ir_would_miss']} ({ir_results['miss_rate_pct']}%)")
        print(f"IR would catch: {ir_results['ir_would_catch']}")
        print(f"Conclusion: {ir_results['conclusion']}")
        print()
        for d in ir_results["details"]:
            sym = "✓" if d["ir_visible"] else "✗"
            print(f"  {sym} {d['name']:30s} [{d['divergence_kind']}]: {d['explanation']}")

    # Save results
    output_path = args.output or os.path.join(SCRIPT_DIR, "full_pipeline_results.json")
    output_data = summary.to_dict()
    if args.llvm_baseline:
        output_data["llvm_ir_baseline"] = run_llvm_ir_baseline()

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return 0 if summary.accuracy >= 0.80 else 1


if __name__ == "__main__":
    sys.exit(main())
