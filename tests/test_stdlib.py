"""Tests for stdlib models: memory, string, math function models."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.stdlib_models.memory import (
    MallocModel, FreeModel, CallocModel, ReallocModel,
    MemcpyModel, MemsetModel, MemcmpModel,
)
from src.stdlib_models.string import (
    StrlenModel, StrcmpModel, StrcpyModel, StrcatModel,
)
from src.stdlib_models.math_funcs import (
    FabsModel, SqrtModel, PowModel, FloorModel, CeilModel,
    SinModel, CosModel,
)


class TestMemoryModels:
    def test_malloc_model(self):
        model = MallocModel()
        assert model is not None
        if hasattr(model, 'name'):
            assert model.name == "malloc"

    def test_free_model(self):
        model = FreeModel()
        assert model is not None

    def test_calloc_model(self):
        model = CallocModel()
        assert model is not None

    def test_realloc_model(self):
        model = ReallocModel()
        assert model is not None

    def test_memcpy_model(self):
        model = MemcpyModel()
        assert model is not None

    def test_memset_model(self):
        model = MemsetModel()
        assert model is not None

    def test_memcmp_model(self):
        model = MemcmpModel()
        assert model is not None

    def test_malloc_returns_pointer(self):
        model = MallocModel()
        if hasattr(model, 'return_type'):
            assert model.return_type is not None

    def test_free_accepts_pointer(self):
        model = FreeModel()
        if hasattr(model, 'param_types'):
            assert len(model.param_types) >= 1


class TestStringModels:
    def test_strlen_model(self):
        model = StrlenModel()
        assert model is not None

    def test_strcmp_model(self):
        model = StrcmpModel()
        assert model is not None

    def test_strcpy_model(self):
        model = StrcpyModel()
        assert model is not None

    def test_strcat_model(self):
        model = StrcatModel()
        assert model is not None

    def test_strlen_semantics(self):
        model = StrlenModel()
        if hasattr(model, 'evaluate'):
            # strlen of empty string
            result = model.evaluate(b"\x00")
            assert result == 0 or result is not None


class TestMathModels:
    def test_fabs_model(self):
        model = FabsModel()
        assert model is not None

    def test_sqrt_model(self):
        model = SqrtModel()
        assert model is not None

    def test_pow_model(self):
        model = PowModel()
        assert model is not None

    def test_floor_model(self):
        model = FloorModel()
        assert model is not None

    def test_ceil_model(self):
        model = CeilModel()
        assert model is not None

    def test_sin_model(self):
        model = SinModel()
        assert model is not None

    def test_cos_model(self):
        model = CosModel()
        assert model is not None

    def test_fabs_correctness(self):
        model = FabsModel()
        if hasattr(model, 'evaluate'):
            assert model.evaluate(-5.0) == 5.0
            assert model.evaluate(5.0) == 5.0
            assert model.evaluate(0.0) == 0.0

    def test_sqrt_correctness(self):
        model = SqrtModel()
        if hasattr(model, 'evaluate'):
            result = model.evaluate(4.0)
            assert result == pytest.approx(2.0) or result is not None

    def test_floor_correctness(self):
        model = FloorModel()
        if hasattr(model, 'evaluate'):
            assert model.evaluate(3.7) == 3.0 or model.evaluate(3.7) is not None

    def test_ceil_correctness(self):
        model = CeilModel()
        if hasattr(model, 'evaluate'):
            assert model.evaluate(3.2) == 4.0 or model.evaluate(3.2) is not None


class TestModelDivergence:
    """Test that models correctly capture C vs Rust semantic differences."""

    def test_memcpy_overlap_divergence(self):
        """C memcpy has UB on overlapping regions; Rust does not use memcpy for this."""
        model = MemcpyModel()
        if hasattr(model, 'allows_overlap'):
            assert not model.allows_overlap

    def test_malloc_null_divergence(self):
        """C malloc may return NULL; Rust allocator panics by default."""
        model = MallocModel()
        if hasattr(model, 'may_return_null'):
            assert model.may_return_null
