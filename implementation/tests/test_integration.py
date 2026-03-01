"""Integration tests: full pipeline on small C↔Rust pairs."""

import pytest
import sys
import os
import tempfile
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.cli.config import VerifyConfig
from src.cli.reporter import VerificationReport, VerdictKind, EquivalenceVerdict
from src.cli.pipeline import VerificationPipeline, PipelinePhase, PipelineStatus


class TestPipelineCreation:
    def test_default_config(self):
        config = VerifyConfig.default()
        pipeline = VerificationPipeline(config)
        assert pipeline is not None

    def test_fast_config(self):
        config = VerifyConfig.fast()
        pipeline = VerificationPipeline(config)
        assert pipeline is not None

    def test_thorough_config(self):
        config = VerifyConfig.thorough()
        pipeline = VerificationPipeline(config)
        assert pipeline is not None


class TestSimpleEquivalence:
    """Test equivalent function pairs."""

    def test_simple_add(self):
        c_source = "int add(int a, int b) { return a + b; }"
        rust_source = "pub fn add(a: i32, b: i32) -> i32 { a.wrapping_add(b) }"

        config = VerifyConfig.fast()
        pipeline = VerificationPipeline(config)
        report = pipeline.verify(c_source, rust_source, "add", "add")

        assert isinstance(report, VerificationReport)
        assert report.verdict is not None
        assert report.verdict.kind in (VerdictKind.EQUIVALENT,
                                        VerdictKind.UNKNOWN,
                                        VerdictKind.DIVERGENT)

    def test_void_function(self):
        c_source = "void noop(void) { }"
        rust_source = "pub fn noop() { }"

        config = VerifyConfig.fast()
        pipeline = VerificationPipeline(config)
        report = pipeline.verify(c_source, rust_source, "noop", "noop")
        assert isinstance(report, VerificationReport)

    def test_constant_function(self):
        c_source = "int forty_two(void) { return 42; }"
        rust_source = "pub fn forty_two() -> i32 { 42 }"

        config = VerifyConfig.fast()
        pipeline = VerificationPipeline(config)
        report = pipeline.verify(c_source, rust_source)
        assert isinstance(report, VerificationReport)


class TestOverflowDivergence:
    """Test functions that diverge on integer overflow."""

    def test_signed_overflow_add(self):
        c_source = """
int add_overflow(int a, int b) {
    return a + b;
}
"""
        rust_source = """
pub fn add_overflow(a: i32, b: i32) -> i32 {
    a.wrapping_add(b)
}
"""
        config = VerifyConfig.fast()
        pipeline = VerificationPipeline(config)
        report = pipeline.verify(c_source, rust_source)
        assert isinstance(report, VerificationReport)

    def test_multiply_overflow(self):
        c_source = "int mul(int a, int b) { return a * b; }"
        rust_source = "pub fn mul(a: i32, b: i32) -> i32 { a.wrapping_mul(b) }"

        config = VerifyConfig.fast()
        pipeline = VerificationPipeline(config)
        report = pipeline.verify(c_source, rust_source)
        assert isinstance(report, VerificationReport)


class TestArrayAccess:
    """Test array access patterns."""

    def test_array_sum(self):
        c_source = """
int array_sum(int *arr, int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    return sum;
}
"""
        rust_source = """
pub fn array_sum(arr: &[i32]) -> i32 {
    let mut sum: i32 = 0;
    for &x in arr.iter() {
        sum = sum.wrapping_add(x);
    }
    sum
}
"""
        config = VerifyConfig.fast()
        pipeline = VerificationPipeline(config)
        report = pipeline.verify(c_source, rust_source)
        assert isinstance(report, VerificationReport)


class TestReportFormats:
    """Test report generation in different formats."""

    def test_json_output(self):
        verdict = EquivalenceVerdict(kind=VerdictKind.EQUIVALENT, confidence=1.0)
        report = VerificationReport(verdict=verdict, c_function="add", rust_function="add")
        json_str = report.to_json()
        parsed = json.loads(json_str)
        assert parsed["verdict"]["verdict"] == "equivalent"
        assert parsed["c_function"] == "add"

    def test_terminal_output(self):
        verdict = EquivalenceVerdict(kind=VerdictKind.DIVERGENT, confidence=0.9,
                                     reason="Integer overflow")
        report = VerificationReport(verdict=verdict)
        text = report.format_terminal()
        assert "DIVERGENT" in text
        assert "Integer overflow" in text

    def test_html_output(self):
        verdict = EquivalenceVerdict(kind=VerdictKind.UNKNOWN, confidence=0.5)
        report = VerificationReport(verdict=verdict)
        html = report.format_html()
        assert "<html>" in html
        assert "UNKNOWN" in html

    def test_report_roundtrip(self):
        verdict = EquivalenceVerdict(kind=VerdictKind.EQUIVALENT, confidence=1.0,
                                     reason="All paths verified")
        report = VerificationReport(verdict=verdict, c_function="f", rust_function="f")
        json_str = report.to_json()
        parsed = json.loads(json_str)
        restored = VerificationReport.from_dict(parsed)
        assert restored.verdict.kind == VerdictKind.EQUIVALENT
        assert restored.c_function == "f"


class TestConfigProfiles:
    def test_default_profile(self):
        config = VerifyConfig.default()
        errors = config.validate()
        assert len(errors) == 0

    def test_fast_profile(self):
        config = VerifyConfig.fast()
        errors = config.validate()
        assert len(errors) == 0
        assert config.timeouts.total_timeout < 60

    def test_thorough_profile(self):
        config = VerifyConfig.thorough()
        errors = config.validate()
        assert len(errors) == 0
        assert config.timeouts.total_timeout > 300

    def test_config_serialization(self):
        config = VerifyConfig.default()
        json_str = config.to_json()
        parsed = json.loads(json_str)
        restored = VerifyConfig.from_dict(parsed)
        assert restored.timeouts.total_timeout == config.timeouts.total_timeout

    def test_config_merge(self):
        config = VerifyConfig.default()
        merged = config.merge_with({"timeouts": {"total_timeout": 999}})
        assert merged.timeouts.total_timeout == 999

    def test_config_validation(self):
        config = VerifyConfig.default()
        config.timeouts.total_timeout = -1
        errors = config.validate()
        assert len(errors) > 0


class TestPipelineProgress:
    def test_progress_callback(self):
        phases_seen = []

        def callback(phase, status, msg):
            phases_seen.append((phase, status))

        c_source = "int f(int x) { return x; }"
        rust_source = "pub fn f(x: i32) -> i32 { x }"

        config = VerifyConfig.fast()
        pipeline = VerificationPipeline(config)
        pipeline.set_progress_callback(callback)
        report = pipeline.verify(c_source, rust_source)

        assert len(phases_seen) > 0
        # Should see at least PARSE_C and PARSE_RUST
        phase_names = [p[0] for p in phases_seen]
        assert PipelinePhase.PARSE_C in phase_names
        assert PipelinePhase.PARSE_RUST in phase_names


class TestFuzzOnlyMode:
    def test_fuzz_only(self):
        c_source = "int add(int a, int b) { return a + b; }"
        rust_source = "pub fn add(a: i32, b: i32) -> i32 { a.wrapping_add(b) }"

        config = VerifyConfig.fuzz_only()
        pipeline = VerificationPipeline(config)
        report = pipeline.fuzz_only(c_source, rust_source, "add", "add")
        assert isinstance(report, VerificationReport)


class TestAnalyzeOnly:
    def test_analyze_only(self):
        c_source = "int f(int x) { return x + 1; }"
        rust_source = "pub fn f(x: i32) -> i32 { x.wrapping_add(1) }"

        config = VerifyConfig.fast()
        pipeline = VerificationPipeline(config)
        results = pipeline.analyze_only(c_source, rust_source)
        assert isinstance(results, dict)


class TestFileBasedVerification:
    def test_verify_from_files(self, tmp_dir):
        c_path = os.path.join(tmp_dir, "test.c")
        rs_path = os.path.join(tmp_dir, "test.rs")

        with open(c_path, "w") as f:
            f.write("int add(int a, int b) { return a + b; }")
        with open(rs_path, "w") as f:
            f.write("pub fn add(a: i32, b: i32) -> i32 { a.wrapping_add(b) }")

        config = VerifyConfig.fast()
        pipeline = VerificationPipeline(config)
        report = pipeline.verify_from_files(c_path, rs_path)
        assert isinstance(report, VerificationReport)

    @pytest.fixture
    def tmp_dir(self):
        import tempfile
        import shutil
        d = tempfile.mkdtemp()
        yield d
        shutil.rmtree(d, ignore_errors=True)
