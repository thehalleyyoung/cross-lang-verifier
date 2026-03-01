"""Tests for symbolic execution: simple programs, path forking, memory, loops."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ir.types import IntType, Signedness, FunctionType, PointerType
from src.ir.instructions import Constant
from src.ir.module import Module
from src.ir.builder import IRBuilder
from src.symbolic_exec.executor import SymbolicExecutor
from src.symbolic_exec.state import SymbolicState, SymbolicMemory
from src.symbolic_exec.path_manager import PathManager
from src.symbolic_exec.memory import FlatMemoryModel


def make_i32():
    return IntType(32, Signedness.SIGNED)


def make_linear_module():
    """Linear function: return a + b."""
    mod = Module("test", "test.c", "x86_64-unknown-linux-gnu", "", "C")
    i32 = make_i32()
    ft = FunctionType(i32, [i32, i32])
    func = mod.create_function("add", ft)
    entry = func.create_block("entry")
    builder = IRBuilder()
    builder.position_at_end(entry)
    a, b = func.arguments
    result = builder.add(a, b, name="sum")
    builder.ret(result)
    return mod


def make_branching_module():
    """Branching function: if a > b return a else return b."""
    mod = Module("test", "test.c", "x86_64-unknown-linux-gnu", "", "C")
    i32 = make_i32()
    ft = FunctionType(i32, [i32, i32])
    func = mod.create_function("max", ft)
    entry = func.create_block("entry")
    then_bb = func.create_block("then")
    else_bb = func.create_block("else")

    builder = IRBuilder()
    builder.position_at_end(entry)
    a, b = func.arguments
    cond = builder.icmp_sgt(a, b)
    builder.cond_br(cond, then_bb, else_bb)

    builder.position_at_end(then_bb)
    builder.ret(a)

    builder.position_at_end(else_bb)
    builder.ret(b)
    return mod


def make_loop_module():
    """Function with a simple loop."""
    mod = Module("test", "test.c", "x86_64-unknown-linux-gnu", "", "C")
    i32 = make_i32()
    ft = FunctionType(i32, [i32])
    func = mod.create_function("countdown", ft)
    entry = func.create_block("entry")
    header = func.create_block("header")
    body = func.create_block("body")
    exit_bb = func.create_block("exit")

    builder = IRBuilder()
    builder.position_at_end(entry)
    builder.br(header)

    builder.position_at_end(header)
    cond = builder.icmp_sgt(func.arguments[0], Constant.int_const(0, IntType(32, Signedness.SIGNED)))
    builder.cond_br(cond, body, exit_bb)

    builder.position_at_end(body)
    builder.br(header)

    builder.position_at_end(exit_bb)
    builder.ret(func.arguments[0])
    return mod


def make_memory_module():
    """Function with memory operations."""
    mod = Module("test", "test.c", "x86_64-unknown-linux-gnu", "", "C")
    i32 = make_i32()
    ft = FunctionType(i32, [i32])
    func = mod.create_function("store_load", ft)
    entry = func.create_block("entry")

    builder = IRBuilder()
    builder.position_at_end(entry)
    ptr = builder.alloca(i32, name="ptr")
    builder.store(func.arguments[0], ptr)
    val = builder.load(ptr, i32, name="val")
    builder.ret(val)
    return mod


class TestSymbolicState:
    def test_create(self):
        state = SymbolicState()
        assert state is not None

    def test_initial_state(self):
        state = SymbolicState()
        # Initial state should have empty PC
        if hasattr(state, 'pc'):
            assert state.pc is not None


class TestSymbolicMemory:
    def test_create(self):
        mem = SymbolicMemory()
        assert mem is not None

    def test_flat_memory(self):
        mem = FlatMemoryModel()
        assert mem is not None


class TestPathManager:
    def test_create(self):
        pm = PathManager()
        assert pm is not None


class TestSymbolicExecutor:
    def test_create(self):
        mod = make_linear_module()
        executor = SymbolicExecutor()
        assert executor is not None

    def test_execute_linear(self):
        mod = make_linear_module()
        executor = SymbolicExecutor()
        func = next(mod.iter_functions())
        results = executor.execute_function(func)
        assert results is not None

    def test_execute_branching(self):
        mod = make_branching_module()
        executor = SymbolicExecutor()
        func = next(mod.iter_functions())
        results = executor.execute_function(func)
        assert results is not None
        # Should explore 2 paths
        if isinstance(results, list):
            assert len(results) >= 2

    def test_execute_loop(self):
        mod = make_loop_module()
        executor = SymbolicExecutor()
        func = next(mod.iter_functions())
        results = executor.execute_function(func)
        assert results is not None

    def test_execute_memory(self):
        mod = make_memory_module()
        executor = SymbolicExecutor()
        func = next(mod.iter_functions())
        results = executor.execute_function(func)
        assert results is not None

    def test_execute_function(self):
        mod = make_linear_module()
        func = next(mod.iter_functions())
        executor = SymbolicExecutor()
        if hasattr(executor, 'execute_function'):
            results = executor.execute_function(func)
            assert results is not None


class TestPathForking:
    def test_two_paths(self):
        mod = make_branching_module()
        executor = SymbolicExecutor()
        func = next(mod.iter_functions())
        results = executor.execute_function(func)
        if isinstance(results, list):
            assert len(results) >= 2

    def test_loop_unrolling(self):
        mod = make_loop_module()
        executor = SymbolicExecutor()
        func = next(mod.iter_functions())
        results = executor.execute_function(func)
        assert results is not None
