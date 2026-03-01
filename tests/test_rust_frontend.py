"""Tests for Rust frontend: lex, parse, and lower unsafe Rust."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.frontend_rust.lexer import RustLexer
from src.frontend_rust.parser import RustParser
from src.frontend_rust.rust_ast import Crate
from src.frontend_rust.ir_lowering import RustIRLowering
from src.frontend_rust.type_resolver import RustTypeResolver


class TestRustLexer:
    def test_empty(self):
        lexer = RustLexer("", "test.rs")
        tokens = lexer.tokenize()
        assert isinstance(tokens, list)

    def test_simple_function(self):
        source = "pub fn add(a: i32, b: i32) -> i32 { a + b }"
        lexer = RustLexer(source, "test.rs")
        tokens = lexer.tokenize()
        assert len(tokens) > 0

    def test_keywords(self):
        source = "fn let mut if else while for loop match return pub struct enum impl unsafe"
        lexer = RustLexer(source, "test.rs")
        tokens = lexer.tokenize()
        assert len(tokens) >= 10

    def test_integer_literals(self):
        source = "42 0xFF 0o77 0b1010 42_u32 1_000_i64"
        lexer = RustLexer(source, "test.rs")
        tokens = lexer.tokenize()
        assert len(tokens) >= 1

    def test_operators(self):
        source = "+ - * / % & | ^ ! << >> == != < > <= >= && || -> =>"
        lexer = RustLexer(source, "test.rs")
        tokens = lexer.tokenize()
        assert len(tokens) >= 10

    def test_string_literal(self):
        source = '"hello world"'
        lexer = RustLexer(source, "test.rs")
        tokens = lexer.tokenize()
        assert len(tokens) >= 1

    def test_comments(self):
        source = """
// line comment
let x: i32 = 1; /* block comment */
let y: i32 = 2;
"""
        lexer = RustLexer(source, "test.rs")
        tokens = lexer.tokenize()
        assert len(tokens) > 0

    def test_type_annotations(self):
        source = "let x: i32 = 0; let y: u64 = 1; let z: f64 = 2.0;"
        lexer = RustLexer(source, "test.rs")
        tokens = lexer.tokenize()
        assert len(tokens) > 0


class TestRustParser:
    def _parse(self, source):
        parser = RustParser(source, "test.rs")
        return parser.parse()

    def test_simple_function(self):
        source = "pub fn add(a: i32, b: i32) -> i32 { a + b }"
        ast = self._parse(source)
        assert isinstance(ast, Crate)

    def test_wrapping_operations(self):
        source = """
pub fn wrapping_add(a: i32, b: i32) -> i32 {
    a.wrapping_add(b)
}
"""
        ast = self._parse(source)
        assert isinstance(ast, Crate)

    def test_checked_operations(self):
        source = """
pub fn checked_add(a: i32, b: i32) -> Option<i32> {
    a.checked_add(b)
}
"""
        ast = self._parse(source)
        assert isinstance(ast, Crate)

    def test_let_bindings(self):
        source = """
pub fn f(x: i32) -> i32 {
    let y: i32 = x + 1;
    let z: i32 = y * 2;
    z
}
"""
        ast = self._parse(source)
        assert isinstance(ast, Crate)

    def test_if_else(self):
        source = """
pub fn max(a: i32, b: i32) -> i32 {
    if a > b { a } else { b }
}
"""
        ast = self._parse(source)
        assert isinstance(ast, Crate)

    def test_while_loop(self):
        source = """
pub fn sum(n: i32) -> i32 {
    let mut s: i32 = 0;
    let mut i: i32 = 0;
    while i < n {
        s = s.wrapping_add(i);
        i = i.wrapping_add(1);
    }
    s
}
"""
        ast = self._parse(source)
        assert isinstance(ast, Crate)

    def test_explicit_types(self):
        source = """
pub fn cast_demo(x: i32) -> u32 {
    x as u32
}
"""
        ast = self._parse(source)
        assert isinstance(ast, Crate)

    def test_raw_pointers(self):
        source = """
pub unsafe fn deref(p: *const i32) -> i32 {
    *p
}
"""
        ast = self._parse(source)
        assert isinstance(ast, Crate)

    def test_unsafe_block(self):
        source = """
pub fn read_ptr(p: *const i32) -> i32 {
    unsafe { *p }
}
"""
        ast = self._parse(source)
        assert isinstance(ast, Crate)

    def test_struct_definition(self):
        source = """
pub struct Point {
    pub x: i32,
    pub y: i32,
}
"""
        ast = self._parse(source)
        assert isinstance(ast, Crate)

    def test_match_expression(self):
        source = """
pub fn classify(x: i32) -> i32 {
    match x {
        0 => 0,
        1 => 1,
        _ => 2,
    }
}
"""
        ast = self._parse(source)
        assert isinstance(ast, Crate)

    def test_function_call(self):
        source = """
fn square(x: i32) -> i32 { x.wrapping_mul(x) }
pub fn sum_squares(a: i32, b: i32) -> i32 {
    square(a).wrapping_add(square(b))
}
"""
        ast = self._parse(source)
        assert isinstance(ast, Crate)

    def test_overflow_methods(self):
        source = """
pub fn overflow_demo(a: i32, b: i32) -> (i32, bool) {
    a.overflowing_add(b)
}
"""
        parser = RustParser(source, "test.rs")
        ast = parser.parse()
        assert isinstance(ast, Crate)


class TestRustTypeResolver:
    def test_create(self):
        resolver = RustTypeResolver()
        assert resolver is not None


class TestRustIRLowering:
    def test_simple_lowering(self):
        source = "pub fn add(a: i32, b: i32) -> i32 { a.wrapping_add(b) }"
        parser = RustParser(source, "test.rs")
        ast = parser.parse()

        lowering = RustIRLowering(RustTypeResolver())
        module = lowering.lower(ast)
        assert module is not None
        assert module.num_functions >= 1

    def test_with_locals(self):
        source = """
pub fn f(x: i32) -> i32 {
    let y: i32 = x.wrapping_add(1);
    y
}
"""
        parser = RustParser(source, "test.rs")
        ast = parser.parse()

        lowering = RustIRLowering(RustTypeResolver())
        module = lowering.lower(ast)
        assert module is not None
