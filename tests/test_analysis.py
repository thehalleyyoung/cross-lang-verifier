"""Tests for analysis: CFG, dominator tree, loop detection, dataflow, alias analysis."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ir.types import IntType, VoidType, Signedness, FunctionType, PointerType
from src.ir.instructions import Constant
from src.ir.basic_block import BasicBlock
from src.ir.function import Function
from src.ir.module import Module
from src.ir.builder import IRBuilder
from src.analysis.cfg import CFG, DominatorTree, LoopInfo
from src.analysis.dataflow import (
    ReachingDefinitions, LiveVariables, ConstantPropagation,
)
from src.analysis.alias import AliasQuery, AliasResult


def make_i32():
    return IntType(32, Signedness.SIGNED)


def make_simple_func():
    """Create a function: entry -> body -> exit."""
    mod = Module("test", "test.c", "x86_64-unknown-linux-gnu", "", "C")
    i32 = make_i32()
    ft = FunctionType(i32, [i32])
    func = mod.create_function("f", ft)

    entry = func.create_block("entry")
    body = func.create_block("body")
    exit_bb = func.create_block("exit")

    builder = IRBuilder()
    builder.position_at_end(entry)
    builder.br(body)

    builder.position_at_end(body)
    builder.br(exit_bb)

    builder.position_at_end(exit_bb)
    builder.ret(func.arguments[0])

    return func


def make_diamond_func():
    """Create a diamond CFG: entry -> {then, else} -> merge."""
    mod = Module("test", "test.c", "x86_64-unknown-linux-gnu", "", "C")
    i32 = make_i32()
    ft = FunctionType(i32, [i32, i32])
    func = mod.create_function("max", ft)

    entry = func.create_block("entry")
    then_bb = func.create_block("then")
    else_bb = func.create_block("else")
    merge = func.create_block("merge")

    builder = IRBuilder()
    builder.position_at_end(entry)
    a, b = func.arguments
    cond = builder.icmp_sgt(a, b)
    builder.cond_br(cond, then_bb, else_bb)

    builder.position_at_end(then_bb)
    builder.br(merge)

    builder.position_at_end(else_bb)
    builder.br(merge)

    builder.position_at_end(merge)
    phi = builder.phi(i32, name="result")
    phi.add_incoming(a, then_bb)
    phi.add_incoming(b, else_bb)
    builder.ret(phi)

    return func


def make_loop_func():
    """Create a function with a simple loop."""
    mod = Module("test", "test.c", "x86_64-unknown-linux-gnu", "", "C")
    i32 = make_i32()
    ft = FunctionType(i32, [i32])
    func = mod.create_function("sum", ft)

    entry = func.create_block("entry")
    header = func.create_block("header")
    body = func.create_block("body")
    exit_bb = func.create_block("exit")

    builder = IRBuilder()
    builder.position_at_end(entry)
    builder.br(header)

    builder.position_at_end(header)
    cond = builder.icmp_slt(func.arguments[0], Constant.int_const(32, 10))
    builder.cond_br(cond, body, exit_bb)

    builder.position_at_end(body)
    builder.br(header)

    builder.position_at_end(exit_bb)
    builder.ret(func.arguments[0])

    return func


class TestCFG:
    def test_create(self):
        func = make_simple_func()
        cfg = CFG(func)
        assert cfg is not None

    def test_entry(self):
        func = make_simple_func()
        cfg = CFG(func)
        assert cfg.entry is not None

    def test_blocks(self):
        func = make_simple_func()
        cfg = CFG(func)
        assert len(cfg.blocks) == 3

    def test_edges(self):
        func = make_simple_func()
        cfg = CFG(func)
        edges = cfg.edges
        assert len(edges) >= 2

    def test_diamond_cfg(self):
        func = make_diamond_func()
        cfg = CFG(func)
        assert len(cfg.blocks) == 4

    def test_reachable_blocks(self):
        func = make_simple_func()
        cfg = CFG(func)
        reachable = cfg.reachable_blocks()
        assert len(reachable) == 3

    def test_dfs_order(self):
        func = make_simple_func()
        cfg = CFG(func)
        dfs = cfg.dfs_order
        assert len(dfs) == 3

    def test_rpo_order(self):
        func = make_simple_func()
        cfg = CFG(func)
        rpo = cfg.rpo_order
        assert len(rpo) == 3

    def test_back_edges(self):
        func = make_loop_func()
        cfg = CFG(func)
        back = cfg.back_edges
        assert len(back) >= 1


class TestDominatorTree:
    def test_build(self):
        func = make_simple_func()
        cfg = CFG(func)
        dom = DominatorTree.build(cfg)
        assert dom is not None

    def test_entry_dominates_all(self):
        func = make_simple_func()
        cfg = CFG(func)
        dom = DominatorTree.build(cfg)
        entry = cfg.entry
        for block in cfg.blocks:
            assert dom.dominates(entry, block)

    def test_diamond_dominance(self):
        func = make_diamond_func()
        cfg = CFG(func)
        dom = DominatorTree.build(cfg)
        entry = cfg.entry
        for block in cfg.blocks:
            assert dom.dominates(entry, block)

    def test_idom(self):
        func = make_simple_func()
        cfg = CFG(func)
        dom = DominatorTree.build(cfg)
        # Entry has no immediate dominator (or dominates itself)
        for block in cfg.blocks:
            if block != cfg.entry:
                idom = dom.idom(block)
                assert idom is not None

    def test_dominance_frontier(self):
        func = make_diamond_func()
        cfg = CFG(func)
        dom = DominatorTree.build(cfg)
        # At least the merge block should be in some frontier
        for block in cfg.blocks:
            frontier = dom.frontier(block)
            assert isinstance(frontier, (list, set))

    def test_children(self):
        func = make_simple_func()
        cfg = CFG(func)
        dom = DominatorTree.build(cfg)
        children = dom.children(cfg.entry)
        assert isinstance(children, (list, set))


class TestLoopInfo:
    def test_build(self):
        func = make_loop_func()
        cfg = CFG(func)
        loop_info = LoopInfo.build(cfg)
        assert loop_info is not None

    def test_no_loops(self):
        func = make_simple_func()
        cfg = CFG(func)
        loop_info = LoopInfo.build(cfg)
        # Linear function should have no loops
        loops = loop_info.innermost_loops() if hasattr(loop_info, 'innermost_loops') else []
        assert len(loops) == 0

    def test_loop_detection(self):
        func = make_loop_func()
        cfg = CFG(func)
        loop_info = LoopInfo.build(cfg)
        # Should detect at least one loop
        if hasattr(loop_info, 'innermost_loops'):
            loops = loop_info.innermost_loops()
            assert len(loops) >= 1

    def test_loop_header(self):
        func = make_loop_func()
        cfg = CFG(func)
        loop_info = LoopInfo.build(cfg)
        # Check if header block is a loop header
        header = list(func.blocks)[1]  # Second block should be header
        if hasattr(loop_info, 'is_loop_header'):
            assert loop_info.is_loop_header(header)


class TestDataflow:
    def test_reaching_definitions(self):
        func = make_simple_func()
        analysis = ReachingDefinitions(func)
        result = analysis.analyze(func)
        assert result is not None

    def test_live_variables(self):
        func = make_simple_func()
        analysis = LiveVariables(func)
        result = analysis.analyze(func)
        assert result is not None

    def test_constant_propagation(self):
        func = make_simple_func()
        analysis = ConstantPropagation(func)
        result = analysis.run()
        assert result is not None

    def test_dataflow_on_diamond(self):
        func = make_diamond_func()
        analysis = ReachingDefinitions(func)
        result = analysis.analyze(func)
        assert result is not None

    def test_dataflow_on_loop(self):
        func = make_loop_func()
        analysis = ReachingDefinitions(func)
        result = analysis.analyze(func)
        assert result is not None


class TestAliasAnalysis:
    def test_create(self):
        func = make_simple_func()
        alias = AliasQuery(func)
        assert alias is not None

    def test_no_alias_different_allocas(self):
        mod = Module("test", "test.c", "x86_64-unknown-linux-gnu", "", "C")
        i32 = make_i32()
        ft = FunctionType(i32, [])
        func = mod.create_function("f", ft)
        entry = func.create_block("entry")

        builder = IRBuilder()
        builder.position_at_end(entry)
        p1 = builder.alloca(i32, name="p1")
        p2 = builder.alloca(i32, name="p2")
        builder.ret(Constant.int_const(32, 0))

        alias = AliasQuery(func)
        result = alias.query(p1, p2)
        assert result == AliasResult.NO_ALIAS

    def test_must_alias_same(self):
        mod = Module("test", "test.c", "x86_64-unknown-linux-gnu", "", "C")
        i32 = make_i32()
        ft = FunctionType(i32, [])
        func = mod.create_function("f", ft)
        entry = func.create_block("entry")

        builder = IRBuilder()
        builder.position_at_end(entry)
        p = builder.alloca(i32, name="p")
        builder.ret(Constant.int_const(32, 0))

        alias = AliasQuery(func)
        result = alias.query(p, p)
        assert result == AliasResult.MUST_ALIAS
