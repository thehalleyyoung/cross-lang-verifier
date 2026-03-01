"""Tests for C frontend: lex, parse, and lower C source code."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.frontend_c.lexer import CLexer
from src.frontend_c.parser import CParser
from src.frontend_c.c_ast import TranslationUnit
from src.frontend_c.ir_lowering import CIRLowering
from src.frontend_c.type_resolver import CTypeResolver


class TestCLexer:
    def test_empty(self):
        lexer = CLexer("", "test.c")
        tokens = lexer.tokenize()
        assert isinstance(tokens, list)

    def test_simple_function(self):
        source = "int add(int a, int b) { return a + b; }"
        lexer = CLexer(source, "test.c")
        tokens = lexer.tokenize()
        assert len(tokens) > 0

    def test_keywords(self):
        source = "int void return if else while for do switch case break continue"
        lexer = CLexer(source, "test.c")
        tokens = lexer.tokenize()
        assert len(tokens) >= 12

    def test_integer_literals(self):
        source = "42 0xFF 077 0b1010"
        lexer = CLexer(source, "test.c")
        tokens = lexer.tokenize()
        assert len(tokens) >= 1

    def test_operators(self):
        source = "+ - * / % & | ^ ~ << >> == != < > <= >= && || ! ++ -- += -= *= /="
        lexer = CLexer(source, "test.c")
        tokens = lexer.tokenize()
        assert len(tokens) >= 10

    def test_string_literal(self):
        source = '"hello world"'
        lexer = CLexer(source, "test.c")
        tokens = lexer.tokenize()
        assert len(tokens) >= 1

    def test_char_literal(self):
        source = "'a' '\\n' '\\0'"
        lexer = CLexer(source, "test.c")
        tokens = lexer.tokenize()
        assert len(tokens) >= 1

    def test_multiline(self):
        source = """
int x = 10;
int y = 20;
int z = x + y;
"""
        lexer = CLexer(source, "test.c")
        tokens = lexer.tokenize()
        assert len(tokens) > 0

    def test_comments(self):
        source = """
// line comment
int x = 1; /* block comment */
/* multi
   line */
int y = 2;
"""
        lexer = CLexer(source, "test.c")
        tokens = lexer.tokenize()
        assert len(tokens) > 0


class TestCParser:
    def test_simple_function(self):
        source = "int add(int a, int b) { return a + b; }"
        parser = CParser(source, "test.c")
        ast = parser.parse()
        assert isinstance(ast, TranslationUnit)

    def test_void_function(self):
        source = "void noop(void) { }"
        parser = CParser(source, "test.c")
        ast = parser.parse()
        assert isinstance(ast, TranslationUnit)

    def test_variable_declaration(self):
        source = "int f(void) { int x = 42; return x; }"
        parser = CParser(source, "test.c")
        ast = parser.parse()
        assert isinstance(ast, TranslationUnit)

    def test_if_else(self):
        source = """
int max(int a, int b) {
    if (a > b) return a;
    else return b;
}
"""
        parser = CParser(source, "test.c")
        ast = parser.parse()
        assert isinstance(ast, TranslationUnit)

    def test_while_loop(self):
        source = """
int sum(int n) {
    int s = 0;
    int i = 0;
    while (i < n) {
        s += i;
        i++;
    }
    return s;
}
"""
        parser = CParser(source, "test.c")
        ast = parser.parse()
        assert isinstance(ast, TranslationUnit)

    def test_for_loop(self):
        source = """
int sum(int n) {
    int s = 0;
    for (int i = 0; i < n; i++) {
        s += i;
    }
    return s;
}
"""
        parser = CParser(source, "test.c")
        ast = parser.parse()
        assert isinstance(ast, TranslationUnit)

    def test_pointer_access(self):
        source = """
int deref(int *p) { return *p; }
void set(int *p, int v) { *p = v; }
"""
        parser = CParser(source, "test.c")
        ast = parser.parse()
        assert isinstance(ast, TranslationUnit)

    def test_struct_access(self):
        source = """
struct Point { int x; int y; };
int get_x(struct Point *p) { return p->x; }
"""
        parser = CParser(source, "test.c")
        ast = parser.parse()
        assert isinstance(ast, TranslationUnit)

    def test_array_access(self):
        source = """
int sum_array(int *arr, int n) {
    int s = 0;
    for (int i = 0; i < n; i++) {
        s += arr[i];
    }
    return s;
}
"""
        parser = CParser(source, "test.c")
        ast = parser.parse()
        assert isinstance(ast, TranslationUnit)

    def test_function_call(self):
        source = """
int square(int x) { return x * x; }
int sum_squares(int a, int b) { return square(a) + square(b); }
"""
        parser = CParser(source, "test.c")
        ast = parser.parse()
        assert isinstance(ast, TranslationUnit)

    def test_cast_expression(self):
        source = """
unsigned int to_unsigned(int x) { return (unsigned int)x; }
"""
        parser = CParser(source, "test.c")
        ast = parser.parse()
        assert isinstance(ast, TranslationUnit)

    def test_integer_promotions(self):
        source = """
int promote(char a, short b) {
    return a + b;
}
"""
        parser = CParser(source, "test.c")
        ast = parser.parse()
        assert isinstance(ast, TranslationUnit)

    def test_ternary(self):
        source = "int abs_val(int x) { return x >= 0 ? x : -x; }"
        parser = CParser(source, "test.c")
        ast = parser.parse()
        assert isinstance(ast, TranslationUnit)


class TestCTypeResolver:
    def test_create_default(self):
        resolver = CTypeResolver()
        assert resolver is not None

    def test_sizeof_int(self):
        resolver = CTypeResolver()
        from src.frontend_c.c_ast import IntCType
        t = IntCType(is_signed=True, is_int=True)
        size = resolver.sizeof(t)
        assert size == 4

    def test_sizeof_char(self):
        resolver = CTypeResolver()
        from src.frontend_c.c_ast import IntCType
        t = IntCType(is_signed=True, is_char=True)
        size = resolver.sizeof(t)
        assert size == 1


class TestCIRLowering:
    def test_simple_lowering(self):
        source = "int add(int a, int b) { return a + b; }"
        parser = CParser(source, "test.c")
        ast = parser.parse()

        lowering = CIRLowering()
        module = lowering.lower(ast)
        assert module is not None
        assert module.num_functions >= 1

    def test_with_locals(self):
        source = """
int f(int x) {
    int y = x + 1;
    return y;
}
"""
        parser = CParser(source, "test.c")
        ast = parser.parse()

        lowering = CIRLowering()
        module = lowering.lower(ast)
        assert module is not None
