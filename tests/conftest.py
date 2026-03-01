"""Pytest fixtures for cross-language equivalence verifier tests."""

from __future__ import annotations

import os
import sys
import tempfile
import shutil
from typing import Optional

import pytest

# Ensure src is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory that is cleaned up after the test."""
    d = tempfile.mkdtemp(prefix="xlev_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sample_c_source():
    """Sample C source code for testing."""
    return """
int add(int a, int b) {
    return a + b;
}

unsigned int unsigned_add(unsigned int a, unsigned int b) {
    return a + b;
}

int array_sum(int *arr, int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    return sum;
}
"""


@pytest.fixture
def sample_rust_source():
    """Sample Rust source code for testing."""
    return """
pub fn add(a: i32, b: i32) -> i32 {
    a.wrapping_add(b)
}

pub fn unsigned_add(a: u32, b: u32) -> u32 {
    a.wrapping_add(b)
}

pub fn array_sum(arr: &[i32]) -> i32 {
    let mut sum: i32 = 0;
    for &x in arr.iter() {
        sum = sum.wrapping_add(x);
    }
    sum
}
"""


@pytest.fixture
def sample_c_overflow():
    """C source with overflow-prone code."""
    return """
int multiply(int a, int b) {
    return a * b;
}
"""


@pytest.fixture
def sample_rust_overflow():
    """Rust source with checked overflow."""
    return """
pub fn multiply(a: i32, b: i32) -> i32 {
    a.wrapping_mul(b)
}
"""


@pytest.fixture
def sample_ir_module():
    """Build a simple IR module for testing."""
    from src.ir import (
        Module, Function, FunctionType, IntType, Signedness,
        IRBuilder, BasicBlock, Constant,
    )

    mod = Module(name="test", source_filename="test.c",
                 target_triple="x86_64-unknown-linux-gnu",
                 data_layout="", language="C")

    i32 = IntType(32, Signedness.SIGNED)
    func_type = FunctionType(return_type=i32, param_types=[i32, i32])
    func = mod.create_function("add", func_type)

    entry = func.create_block("entry")
    builder = IRBuilder()
    builder.position_at_end(entry)

    a = func.arguments[0]
    b = func.arguments[1]
    result = builder.add(a, b, name="result")
    builder.ret(result)

    return mod


@pytest.fixture
def sample_ir_function(sample_ir_module):
    """Return the first function from the sample IR module."""
    return next(sample_ir_module.iter_functions())


@pytest.fixture
def i32_type():
    from src.ir import IntType, Signedness
    return IntType(32, Signedness.SIGNED)


@pytest.fixture
def u32_type():
    from src.ir import IntType, Signedness
    return IntType(32, Signedness.UNSIGNED)


@pytest.fixture
def i64_type():
    from src.ir import IntType, Signedness
    return IntType(64, Signedness.SIGNED)


@pytest.fixture
def f32_type():
    from src.ir import FloatType, FloatKind
    return FloatType(FloatKind.F32)


@pytest.fixture
def f64_type():
    from src.ir import FloatType, FloatKind
    return FloatType(FloatKind.F64)


@pytest.fixture
def void_type():
    from src.ir import VoidType
    return VoidType()


@pytest.fixture
def write_source_file(tmp_dir):
    """Helper to write source code to a temp file."""
    def _write(content: str, filename: str) -> str:
        path = os.path.join(tmp_dir, filename)
        with open(path, "w") as f:
            f.write(content)
        return path
    return _write
