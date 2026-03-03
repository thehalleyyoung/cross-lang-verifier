"""Verification pipeline: orchestrates the full equivalence verification workflow.

Steps:
1. Parse both C and Rust sources
2. Lower to shared IR
3. Run analysis passes (CFG, dominators, alias analysis)
4. Construct product program with alignment and coercions
5. Run symbolic execution on product program
6. Check divergence conditions via SMT solver
7. Run fuzzer on timed-out / unknown paths
8. Generate verification report
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Callable

from .config import VerifyConfig
from .reporter import (
    EquivalenceVerdict, VerdictKind, VerificationReport,
    Counterexample, ConcreteValue, CoverageSummary,
    DivergenceSummary, DivergenceCategory, TimingInfo,
)

logger = logging.getLogger(__name__)


class PipelinePhase(Enum):
    """Phases of the verification pipeline."""
    INIT = "init"
    PARSE_C = "parse_c"
    PARSE_RUST = "parse_rust"
    LOWER_C = "lower_c"
    LOWER_RUST = "lower_rust"
    VALIDATE_IR = "validate_ir"
    ANALYSIS = "analysis"
    NORMALIZE = "normalize"
    ALIGN = "align"
    PRODUCT = "product"
    SYMBOLIC = "symbolic"
    SMT = "smt"
    FUZZ = "fuzz"
    REPORT = "report"
    DONE = "done"


class PipelineStatus(Enum):
    """Status of a pipeline phase."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMED_OUT = "timed_out"


@dataclass
class PhaseResult:
    """Result of a single pipeline phase."""
    phase: PipelinePhase
    status: PipelineStatus
    duration: float = 0.0
    error: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineState:
    """Mutable state carried through the pipeline."""
    c_source: str = ""
    rust_source: str = ""
    c_ast: Any = None
    rust_ast: Any = None
    c_ir: Any = None
    rust_ir: Any = None
    c_cfg: Any = None
    rust_cfg: Any = None
    c_dom_tree: Any = None
    rust_dom_tree: Any = None
    c_loop_info: Any = None
    rust_loop_info: Any = None
    alias_info_c: Any = None
    alias_info_rust: Any = None
    alignment: Any = None
    product_program: Any = None
    symbolic_results: List[Any] = field(default_factory=list)
    smt_results: List[Any] = field(default_factory=list)
    fuzz_results: Any = None
    counterexamples: List[Counterexample] = field(default_factory=list)
    divergence_summary: DivergenceSummary = field(default_factory=DivergenceSummary)
    coverage: CoverageSummary = field(default_factory=CoverageSummary)
    warnings: List[str] = field(default_factory=list)
    phase_results: List[PhaseResult] = field(default_factory=list)


ProgressCallback = Callable[[PipelinePhase, PipelineStatus, str], None]


class VerificationPipeline:
    """Orchestrates the full cross-language equivalence verification workflow."""

    def __init__(self, config: Optional[VerifyConfig] = None):
        self.config = config or VerifyConfig.default()
        self._progress_callback: Optional[ProgressCallback] = None
        self._state = PipelineState()
        self._start_time = 0.0
        self._phase_handlers: Dict[PipelinePhase, Callable] = {
            PipelinePhase.PARSE_C: self._phase_parse_c,
            PipelinePhase.PARSE_RUST: self._phase_parse_rust,
            PipelinePhase.LOWER_C: self._phase_lower_c,
            PipelinePhase.LOWER_RUST: self._phase_lower_rust,
            PipelinePhase.VALIDATE_IR: self._phase_validate_ir,
            PipelinePhase.ANALYSIS: self._phase_analysis,
            PipelinePhase.NORMALIZE: self._phase_normalize,
            PipelinePhase.ALIGN: self._phase_align,
            PipelinePhase.PRODUCT: self._phase_product,
            PipelinePhase.SYMBOLIC: self._phase_symbolic,
            PipelinePhase.SMT: self._phase_smt,
            PipelinePhase.FUZZ: self._phase_fuzz,
            PipelinePhase.REPORT: self._phase_report,
        }

    def set_progress_callback(self, callback: ProgressCallback) -> None:
        self._progress_callback = callback

    def _notify(self, phase: PipelinePhase, status: PipelineStatus, msg: str = "") -> None:
        if self._progress_callback:
            self._progress_callback(phase, status, msg)
        level = logging.DEBUG if status == PipelineStatus.COMPLETED else logging.INFO
        logger.log(level, "Phase %s: %s %s", phase.value, status.value, msg)

    def _elapsed(self) -> float:
        return time.time() - self._start_time

    def _check_timeout(self) -> bool:
        return self._elapsed() > self.config.timeouts.total_timeout

    def _remaining_budget(self) -> float:
        """Return remaining time budget in seconds."""
        return max(0.0, self.config.timeouts.total_timeout - self._elapsed())

    def _run_phase(self, phase: PipelinePhase) -> PhaseResult:
        handler = self._phase_handlers.get(phase)
        if handler is None:
            return PhaseResult(phase, PipelineStatus.SKIPPED)

        self._notify(phase, PipelineStatus.RUNNING)
        t0 = time.time()
        try:
            if self._check_timeout():
                result = PhaseResult(phase, PipelineStatus.TIMED_OUT,
                                     duration=time.time() - t0,
                                     error="Total timeout exceeded")
                self._state.warnings.append(f"Phase {phase.value} skipped: total timeout")
                self._notify(phase, PipelineStatus.TIMED_OUT)
                return result

            data = handler()
            duration = time.time() - t0
            result = PhaseResult(phase, PipelineStatus.COMPLETED,
                                 duration=duration, data=data or {})
            self._notify(phase, PipelineStatus.COMPLETED, f"({duration:.2f}s)")
            self._state.phase_results.append(result)
            return result
        except Exception as e:
            duration = time.time() - t0
            result = PhaseResult(phase, PipelineStatus.FAILED,
                                 duration=duration, error=str(e))
            self._state.phase_results.append(result)
            self._notify(phase, PipelineStatus.FAILED, str(e))
            logger.exception("Phase %s failed", phase.value)
            return result

    def verify(self, c_source: str, rust_source: str,
               c_function: Optional[str] = None,
               rust_function: Optional[str] = None) -> VerificationReport:
        """Run the full verification pipeline.

        Args:
            c_source: Path to C source file or C source code string.
            rust_source: Path to Rust source file or Rust source code string.
            c_function: Name of C function to verify (optional).
            rust_function: Name of Rust function to verify (optional).

        Returns:
            VerificationReport with verdict, counterexamples, and statistics.
        """
        self._start_time = time.time()
        self._state = PipelineState()
        self._state.c_source = c_source
        self._state.rust_source = rust_source

        if c_function:
            self.config.c_function = c_function
        if rust_function:
            self.config.rust_function = rust_function

        phases = [
            PipelinePhase.PARSE_C,
            PipelinePhase.PARSE_RUST,
            PipelinePhase.LOWER_C,
            PipelinePhase.LOWER_RUST,
            PipelinePhase.VALIDATE_IR,
            PipelinePhase.ANALYSIS,
            PipelinePhase.NORMALIZE,
            PipelinePhase.ALIGN,
            PipelinePhase.PRODUCT,
            PipelinePhase.SYMBOLIC,
            PipelinePhase.SMT,
            PipelinePhase.FUZZ,
            PipelinePhase.REPORT,
        ]

        skip_product_chain = False
        for phase in phases:
            # Skip product-dependent phases if we fell back to structural
            if skip_product_chain and phase in (
                PipelinePhase.SYMBOLIC, PipelinePhase.SMT,
            ):
                self._state.warnings.append(
                    f"Phase {phase.value} skipped: using structural fallback")
                continue

            result = self._run_phase(phase)
            if result.status == PipelineStatus.FAILED:
                if phase in (PipelinePhase.PARSE_C, PipelinePhase.PARSE_RUST,
                             PipelinePhase.LOWER_C, PipelinePhase.LOWER_RUST):
                    return self._build_error_report(result)
                # Non-critical phases: log warning and continue
                self._state.warnings.append(f"Phase {phase.value} failed: {result.error}")
                # Alignment/product failures: try structural comparison fallback
                if phase in (PipelinePhase.ALIGN, PipelinePhase.PRODUCT):
                    self._state.warnings.append(
                        "Falling back to structural comparison")
                    self._try_structural_comparison()
                    skip_product_chain = True

        return self._build_report()

    def verify_from_files(self, c_path: str, rust_path: str,
                          c_function: Optional[str] = None,
                          rust_function: Optional[str] = None) -> VerificationReport:
        """Convenience: verify from file paths."""
        with open(c_path, "r") as f:
            c_source = f.read()
        with open(rust_path, "r") as f:
            rust_source = f.read()
        return self.verify(c_source, rust_source, c_function, rust_function)

    def fuzz_only(self, c_source: str, rust_source: str,
                  c_function: Optional[str] = None,
                  rust_function: Optional[str] = None) -> VerificationReport:
        """Run only the fuzzing phase (skip symbolic execution and SMT)."""
        self._start_time = time.time()
        self._state = PipelineState()
        self._state.c_source = c_source
        self._state.rust_source = rust_source

        if c_function:
            self.config.c_function = c_function
        if rust_function:
            self.config.rust_function = rust_function

        for phase in [PipelinePhase.PARSE_C, PipelinePhase.PARSE_RUST,
                      PipelinePhase.LOWER_C, PipelinePhase.LOWER_RUST,
                      PipelinePhase.VALIDATE_IR, PipelinePhase.ANALYSIS,
                      PipelinePhase.NORMALIZE, PipelinePhase.ALIGN,
                      PipelinePhase.PRODUCT, PipelinePhase.FUZZ,
                      PipelinePhase.REPORT]:
            result = self._run_phase(phase)
            if result.status == PipelineStatus.FAILED:
                if phase in (PipelinePhase.PARSE_C, PipelinePhase.PARSE_RUST,
                             PipelinePhase.LOWER_C, PipelinePhase.LOWER_RUST):
                    return self._build_error_report(result)

        return self._build_report()

    def analyze_only(self, c_source: str, rust_source: str) -> Dict[str, Any]:
        """Run only analysis passes and return analysis results."""
        self._start_time = time.time()
        self._state = PipelineState()
        self._state.c_source = c_source
        self._state.rust_source = rust_source

        for phase in [PipelinePhase.PARSE_C, PipelinePhase.PARSE_RUST,
                      PipelinePhase.LOWER_C, PipelinePhase.LOWER_RUST,
                      PipelinePhase.VALIDATE_IR, PipelinePhase.ANALYSIS]:
            result = self._run_phase(phase)
            if result.status == PipelineStatus.FAILED:
                return {"error": result.error, "phase": phase.value}

        return {
            "c_cfg": repr(self._state.c_cfg) if self._state.c_cfg else None,
            "rust_cfg": repr(self._state.rust_cfg) if self._state.rust_cfg else None,
            "warnings": self._state.warnings,
            "elapsed": self._elapsed(),
        }

    # -----------------------------------------------------------------------
    # Phase implementations
    # -----------------------------------------------------------------------

    def _phase_parse_c(self) -> Dict[str, Any]:
        """Parse C source into AST with fallback to lenient mode."""
        source = self._state.c_source
        # Try strict parsing first, fall back to lenient
        for attempt, strict in enumerate([True, False]):
            try:
                from ..frontend_c import CLexer, CParser
                parser = CParser(source, filename="<c_source>")
                if not strict:
                    parser.lenient = True
                self._state.c_ast = parser.parse()
                if not strict:
                    self._state.warnings.append("C parsing used lenient mode")
                return {}
            except ImportError:
                self._state.warnings.append("C frontend not fully available")
                return {}
            except Exception as e:
                if strict:
                    continue
                raise RuntimeError(f"C parsing failed: {e}") from e
        return {}

    def _phase_parse_rust(self) -> Dict[str, Any]:
        """Parse Rust source into AST with fallback to lenient mode."""
        source = self._state.rust_source
        for attempt, strict in enumerate([True, False]):
            try:
                from ..frontend_rust import RustLexer, RustParser
                parser = RustParser(source, filename="<rust_source>")
                if not strict:
                    parser.lenient = True
                self._state.rust_ast = parser.parse()
                if not strict:
                    self._state.warnings.append("Rust parsing used lenient mode")
                return {}
            except ImportError:
                self._state.warnings.append("Rust frontend not fully available")
                return {}
            except Exception as e:
                if strict:
                    continue
                raise RuntimeError(f"Rust parsing failed: {e}") from e
        return {}

    def _phase_lower_c(self) -> Dict[str, Any]:
        """Lower C AST to shared IR."""
        if self._state.c_ast is None:
            return {}
        try:
            from ..frontend_c import CIRLowering
            lowering = CIRLowering()
            self._state.c_ir = lowering.lower(self._state.c_ast)
            return {"functions": self._state.c_ir.num_functions if self._state.c_ir else 0}
        except ImportError:
            self._state.warnings.append("C IR lowering not fully available")
            return {}
        except Exception as e:
            raise RuntimeError(f"C IR lowering failed: {e}") from e

    def _phase_lower_rust(self) -> Dict[str, Any]:
        """Lower Rust AST to shared IR."""
        if self._state.rust_ast is None:
            return {}
        try:
            from ..frontend_rust import RustIRLowering
            from ..frontend_rust.type_resolver import RustTypeResolver
            lowering = RustIRLowering(RustTypeResolver())
            self._state.rust_ir = lowering.lower(self._state.rust_ast)
            return {"functions": self._state.rust_ir.num_functions if self._state.rust_ir else 0}
        except ImportError:
            self._state.warnings.append("Rust IR lowering not fully available")
            return {}
        except Exception as e:
            raise RuntimeError(f"Rust IR lowering failed: {e}") from e

    def _phase_validate_ir(self) -> Dict[str, Any]:
        """Validate the IR modules."""
        errors: List[str] = []
        try:
            from ..ir import IRValidator
            validator = IRValidator(strict=True)

            if self._state.c_ir is not None:
                result = validator.validate_module(self._state.c_ir)
                if not result.is_valid:
                    for msg in result.errors:
                        errors.append(f"C IR: {msg}")
                        self._state.warnings.append(f"C IR validation: {msg}")

            if self._state.rust_ir is not None:
                result = validator.validate_module(self._state.rust_ir)
                if not result.is_valid:
                    for msg in result.errors:
                        errors.append(f"Rust IR: {msg}")
                        self._state.warnings.append(f"Rust IR validation: {msg}")
        except ImportError:
            pass

        return {"validation_errors": len(errors)}

    def _phase_analysis(self) -> Dict[str, Any]:
        """Run analysis passes: CFG, dominators, loops, alias analysis, callgraph."""
        stats: Dict[str, Any] = {}
        try:
            from ..analysis import CFG, DominatorTree, LoopInfo, AliasQuery

            if self._state.c_ir is not None:
                for func in self._state.c_ir.iter_functions():
                    cfg = CFG(func)
                    self._state.c_cfg = cfg
                    self._state.c_dom_tree = DominatorTree.build(cfg)
                    self._state.c_loop_info = LoopInfo.build(cfg)
                    if self.config.analysis.run_alias_analysis:
                        self._state.alias_info_c = AliasQuery(
                            func, field_sensitive=self.config.analysis.field_sensitive_alias
                        )
                    break  # Analyze first/target function

            if self._state.rust_ir is not None:
                for func in self._state.rust_ir.iter_functions():
                    cfg = CFG(func)
                    self._state.rust_cfg = cfg
                    self._state.rust_dom_tree = DominatorTree.build(cfg)
                    self._state.rust_loop_info = LoopInfo.build(cfg)
                    if self.config.analysis.run_alias_analysis:
                        self._state.alias_info_rust = AliasQuery(
                            func, field_sensitive=self.config.analysis.field_sensitive_alias
                        )
                    break

            # Interprocedural analysis: build callgraphs for multi-function modules
            if self.config.analysis.interprocedural:
                try:
                    from ..analysis.callgraph import CallGraph
                    if self._state.c_ir is not None:
                        c_cg = CallGraph.build(self._state.c_ir)
                        stats["c_callgraph_nodes"] = len(c_cg)
                    if self._state.rust_ir is not None:
                        r_cg = CallGraph.build(self._state.rust_ir)
                        stats["rust_callgraph_nodes"] = len(r_cg)
                except Exception as e:
                    self._state.warnings.append(f"Interprocedural analysis skipped: {e}")
        except ImportError:
            self._state.warnings.append("Analysis module not fully available")

        return stats

    def _phase_normalize(self) -> Dict[str, Any]:
        """Normalize IR for better alignment."""
        try:
            from ..product_program import IRNormalizer
            normalizer = IRNormalizer()
            if self._state.c_ir is not None:
                normalizer.normalize(self._state.c_ir)
            if self._state.rust_ir is not None:
                normalizer.normalize(self._state.rust_ir)
        except (ImportError, Exception) as e:
            self._state.warnings.append(f"Normalization skipped: {e}")
        return {}

    def _phase_align(self) -> Dict[str, Any]:
        """Align C and Rust functions for product program construction."""
        if self._state.c_ir is None or self._state.rust_ir is None:
            self._state.warnings.append("Alignment skipped: missing IR")
            return {}
        try:
            from ..product_program import FunctionAligner
            c_func = next(self._state.c_ir.iter_functions(), None)
            rust_func = next(self._state.rust_ir.iter_functions(), None)
            if c_func and rust_func:
                aligner = FunctionAligner(c_func, rust_func, self.config)
                self._state.alignment = aligner.align()
                return {"aligned_blocks": len(self._state.alignment.block_alignments)
                        if hasattr(self._state.alignment, 'block_alignments') else 0}
        except (ImportError, Exception) as e:
            self._state.warnings.append(f"Alignment failed: {e}")
            # Fall back to structural comparison
            self._state.warnings.append("Falling back to structural comparison")
            self._try_structural_comparison()
        return {}

    def _phase_product(self) -> Dict[str, Any]:
        """Construct the product program."""
        if self._state.alignment is None:
            self._state.warnings.append("Product construction skipped: no alignment")
            return {}
        try:
            from ..product_program import ProductBuilder
            c_func = next(self._state.c_ir.iter_functions(), None)
            rust_func = next(self._state.rust_ir.iter_functions(), None)
            if c_func and rust_func:
                builder = ProductBuilder(c_func, rust_func, self._state.alignment)
                self._state.product_program = builder.build()
        except (ImportError, Exception) as e:
            self._state.warnings.append(f"Product construction failed: {e}")
            # Try direct SMT comparison as fallback
            self._try_direct_smt_comparison()
        return {}

    def _try_direct_smt_comparison(self) -> None:
        """Fall back to direct SMT comparison when product construction fails."""
        if self._state.c_ir is None or self._state.rust_ir is None:
            return
        try:
            from ..smt.encoder import SMTEncoder, EncodingContext
            from ..semantics.semantic_config import SemanticConfig
            import z3

            c_func = next(self._state.c_ir.iter_functions(), None)
            r_func = next(self._state.rust_ir.iter_functions(), None)
            if c_func is None or r_func is None:
                return

            c_config = SemanticConfig.c11()
            r_config = SemanticConfig.rust_release()
            c_args = list(c_func.arguments)
            r_args = list(r_func.arguments)
            min_args = min(len(c_args), len(r_args))

            shared_vars = []
            c_input_map: Dict[str, Any] = {}
            r_input_map: Dict[str, Any] = {}
            dummy = SMTEncoder(config=c_config)

            for i in range(min_args):
                sort = z3.BitVecSort(32)
                try:
                    if c_args[i].type:
                        sort = dummy.encode_type(c_args[i].type)
                except Exception:
                    pass
                try:
                    v = z3.BitVec(f"input_{i}", sort.size()) if z3.is_bv_sort(sort) \
                        else z3.Const(f"input_{i}", sort)
                except Exception:
                    v = z3.BitVec(f"input_{i}", 32)
                shared_vars.append((f"input_{i}", v))
                ca = c_args[i].name or f"arg_{c_args[i].index}"
                ra = r_args[i].name or f"arg_{r_args[i].index}"
                c_input_map[ca] = v
                r_input_map[ra] = v

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
                return

            solver = z3.Solver()
            budget_ms = int(self._remaining_budget() * 1000)
            solver.set("timeout", min(budget_ms, 5000) if budget_ms > 0 else 5000)
            for a in c_ctx.assertions:
                solver.add(a)
            for a in r_ctx.assertions:
                solver.add(a)

            c_r, r_r = c_ret, r_ret
            if z3.is_bv(c_r) and z3.is_bv(r_r):
                c_w, r_w = c_r.size(), r_r.size()
                if c_w != r_w:
                    tw = max(c_w, r_w)
                    if c_w < tw:
                        c_r = z3.SignExt(tw - c_w, c_r)
                    if r_w < tw:
                        r_r = z3.SignExt(tw - r_w, r_r)
            solver.add(c_r != r_r)

            result = solver.check()
            if result == z3.unsat:
                self._state.coverage.verified_paths = 1
                self._state.coverage.total_paths = 1
            elif result == z3.sat:
                m = solver.model()
                from ..cli.reporter import Counterexample as RCex, ConcreteValue, DivergenceCategory
                inputs = []
                for nm, var in shared_vars:
                    val = m.evaluate(var, model_completion=True)
                    inputs.append(ConcreteValue(name=nm, c_value=str(val), rust_value=str(val)))
                self._state.counterexamples.append(RCex(
                    inputs=inputs,
                    category=DivergenceCategory.OTHER,
                    description="Found by direct SMT comparison (product fallback)",
                ))
                self._state.divergence_summary.add(DivergenceCategory.OTHER)
            self._state.warnings.append("Used direct SMT comparison fallback")
        except Exception as e:
            self._state.warnings.append(f"Direct SMT fallback failed: {e}")

    def _phase_symbolic(self) -> Dict[str, Any]:
        """Run symbolic execution on the product program."""
        if self._state.product_program is None:
            self._state.warnings.append("Symbolic execution skipped: no product program")
            return {}
        if self.config.symbolic.max_paths <= 0:
            return {"skipped": True}
        try:
            from ..symbolic_exec import SymbolicExecutor
            executor = SymbolicExecutor(self._state.product_program, self.config)
            results = executor.execute()
            self._state.symbolic_results = results if isinstance(results, list) else [results]
            return {"paths_explored": len(self._state.symbolic_results)}
        except (ImportError, Exception) as e:
            self._state.warnings.append(f"Symbolic execution failed: {e}")
        return {}

    def _phase_smt(self) -> Dict[str, Any]:
        """Check divergence conditions via SMT solver."""
        if not self._state.symbolic_results:
            return {}
        try:
            from ..smt import SMTSolver, SMTEncoder, ModelDecoder
            from ..semantics import SemanticConfig

            solver = SMTSolver(self.config)
            encoder = SMTEncoder(SemanticConfig.c11())

            verified = 0
            divergent = 0
            unknown = 0

            for sym_result in self._state.symbolic_results:
                try:
                    formula = encoder.encode_assertion(sym_result)
                    check = solver.check_sat(formula)
                    if hasattr(check, 'value'):
                        if check.value == "sat":
                            divergent += 1
                            model = solver.get_model()
                            if model:
                                decoder = ModelDecoder(model)
                                ce = decoder.extract_counterexample()
                                if ce:
                                    self._state.counterexamples.append(ce)
                        elif check.value == "unsat":
                            verified += 1
                        else:
                            unknown += 1
                except Exception:
                    unknown += 1

            self._state.coverage.verified_paths = verified
            return {"verified": verified, "divergent": divergent, "unknown": unknown}
        except (ImportError, Exception) as e:
            self._state.warnings.append(f"SMT checking failed: {e}")
        return {}

    def _phase_fuzz(self) -> Dict[str, Any]:
        """Run fuzzer on timed-out or unknown paths."""
        if not self.config.fuzzer.enabled:
            return {"skipped": True}
        try:
            from ..fuzzer import FuzzEngine
            engine = FuzzEngine(self._state.product_program, self.config)
            self._state.fuzz_results = engine.run()

            if self._state.fuzz_results and hasattr(self._state.fuzz_results, 'divergences'):
                for div in self._state.fuzz_results.divergences:
                    ce = Counterexample(
                        inputs=[ConcreteValue(name=f"arg{i}", c_value=v, rust_value=v)
                                for i, v in enumerate(getattr(div, 'inputs', []))],
                        c_output=getattr(div, 'c_output', None),
                        rust_output=getattr(div, 'rust_output', None),
                        category=DivergenceCategory.OTHER,
                        description="Found by fuzzer",
                    )
                    self._state.counterexamples.append(ce)
                    self._state.divergence_summary.add(DivergenceCategory.OTHER)

            return {"fuzz_iterations": getattr(self._state.fuzz_results, 'iterations', 0)}
        except (ImportError, Exception) as e:
            self._state.warnings.append(f"Fuzzing failed: {e}")
        return {}

    def _phase_report(self) -> Dict[str, Any]:
        """Finalize the report (compute summary statistics)."""
        total_paths = len(self._state.symbolic_results) + (
            getattr(self._state.fuzz_results, 'paths_tested', 0)
            if self._state.fuzz_results else 0
        )
        self._state.coverage.total_paths = max(total_paths, 1)
        self._state.coverage.explored_paths = total_paths
        return {}

    # -----------------------------------------------------------------------
    # Structural comparison fallback
    # -----------------------------------------------------------------------

    def _try_structural_comparison(self) -> None:
        """Structural comparison fallback when product construction fails.

        Compares C and Rust IR functions by instruction counts, opcode
        histograms, and return type compatibility to produce a partial verdict.
        Results are stored in ``self._state`` for ``_determine_verdict()``.
        """
        if self._state.c_ir is None or self._state.rust_ir is None:
            return
        try:
            c_func = next(self._state.c_ir.iter_functions(), None)
            r_func = next(self._state.rust_ir.iter_functions(), None)
            if c_func is None or r_func is None:
                return

            c_instr_count = c_func.instruction_count
            r_instr_count = r_func.instruction_count
            c_block_count = c_func.num_blocks
            r_block_count = r_func.num_blocks

            # Build opcode histograms
            c_opcodes: Dict[str, int] = {}
            for inst in c_func.iter_instructions():
                op = inst.opcode_name()
                c_opcodes[op] = c_opcodes.get(op, 0) + 1

            r_opcodes: Dict[str, int] = {}
            for inst in r_func.iter_instructions():
                op = inst.opcode_name()
                r_opcodes[op] = r_opcodes.get(op, 0) + 1

            # Jaccard similarity on opcode sets
            all_ops = set(c_opcodes) | set(r_opcodes)
            if all_ops:
                matching = sum(
                    min(c_opcodes.get(op, 0), r_opcodes.get(op, 0))
                    for op in all_ops
                )
                total = sum(
                    max(c_opcodes.get(op, 0), r_opcodes.get(op, 0))
                    for op in all_ops
                )
                opcode_sim = matching / total if total else 0.0
            else:
                opcode_sim = 1.0

            # Instruction count similarity
            max_instr = max(c_instr_count, r_instr_count, 1)
            instr_sim = 1.0 - abs(c_instr_count - r_instr_count) / max_instr

            # Return type compatibility
            ret_compat = 1.0
            try:
                c_ret = c_func.return_type
                r_ret = r_func.return_type
                if c_ret != r_ret:
                    ret_compat = 0.5
                if (hasattr(c_ret, 'is_void') and c_ret.is_void) != \
                   (hasattr(r_ret, 'is_void') and r_ret.is_void):
                    ret_compat = 0.0
            except Exception:
                ret_compat = 0.5

            similarity = 0.4 * opcode_sim + 0.4 * instr_sim + 0.2 * ret_compat

            self._state.phase_results.append(PhaseResult(
                PipelinePhase.PRODUCT, PipelineStatus.COMPLETED,
                data={
                    "structural_similarity": similarity,
                    "c_instr_count": c_instr_count,
                    "r_instr_count": r_instr_count,
                    "c_block_count": c_block_count,
                    "r_block_count": r_block_count,
                    "opcode_similarity": opcode_sim,
                    "fallback": "structural",
                },
            ))
        except Exception as e:
            self._state.warnings.append(f"Structural comparison failed: {e}")

    # -----------------------------------------------------------------------
    # Report builders
    # -----------------------------------------------------------------------

    def _build_report(self) -> VerificationReport:
        """Build the final verification report from pipeline state."""
        verdict = self._determine_verdict()

        timing = TimingInfo(total_seconds=self._elapsed())
        for pr in self._state.phase_results:
            if pr.phase == PipelinePhase.PARSE_C:
                timing.parse_seconds += pr.duration
            elif pr.phase == PipelinePhase.PARSE_RUST:
                timing.parse_seconds += pr.duration
            elif pr.phase == PipelinePhase.ANALYSIS:
                timing.analysis_seconds = pr.duration
            elif pr.phase in (PipelinePhase.ALIGN, PipelinePhase.PRODUCT):
                timing.product_seconds += pr.duration
            elif pr.phase == PipelinePhase.SYMBOLIC:
                timing.symbolic_seconds = pr.duration
            elif pr.phase == PipelinePhase.SMT:
                timing.smt_seconds = pr.duration
            elif pr.phase == PipelinePhase.FUZZ:
                timing.fuzz_seconds = pr.duration

        report = VerificationReport(
            verdict=verdict,
            c_source=self._state.c_source[:200] if len(self._state.c_source) > 200 else self._state.c_source,
            rust_source=self._state.rust_source[:200] if len(self._state.rust_source) > 200 else self._state.rust_source,
            c_function=self.config.c_function,
            rust_function=self.config.rust_function,
            counterexamples=self._state.counterexamples,
            coverage=self._state.coverage,
            divergence_summary=self._state.divergence_summary,
            timing=timing,
            warnings=self._state.warnings,
        )
        return report

    def _build_error_report(self, failed_phase: PhaseResult) -> VerificationReport:
        """Build an error report when an early phase fails."""
        verdict = EquivalenceVerdict(
            kind=VerdictKind.UNKNOWN,
            confidence=0.0,
            reason=f"Pipeline failed at {failed_phase.phase.value}: {failed_phase.error}",
        )
        return VerificationReport(
            verdict=verdict,
            c_source=self._state.c_source[:200],
            rust_source=self._state.rust_source[:200],
            warnings=self._state.warnings + [f"Failed: {failed_phase.error}"],
            timing=TimingInfo(total_seconds=self._elapsed()),
        )

    def _determine_verdict(self) -> EquivalenceVerdict:
        """Determine the final equivalence verdict from pipeline results."""
        if self._state.counterexamples:
            categories = set()
            for ce in self._state.counterexamples:
                categories.add(ce.category.value)
            return EquivalenceVerdict(
                kind=VerdictKind.DIVERGENT,
                confidence=1.0 if any(ce.minimized for ce in self._state.counterexamples) else 0.9,
                reason=f"Found {len(self._state.counterexamples)} counterexample(s) "
                       f"in categories: {', '.join(sorted(categories))}",
            )

        verified = self._state.coverage.verified_paths
        total = self._state.coverage.total_paths

        # Check for structural similarity info from fallback
        structural_sim = None
        for pr in self._state.phase_results:
            if pr.data.get("fallback") == "structural":
                structural_sim = pr.data.get("structural_similarity", 0.0)

        if verified > 0 and verified == total:
            confidence = 0.95 if total <= 1 else 1.0
            return EquivalenceVerdict(
                kind=VerdictKind.EQUIVALENT,
                confidence=confidence,
                reason=f"All {verified} paths verified equivalent by SMT solver",
            )

        if verified > 0:
            confidence = verified / total if total > 0 else 0.5
            return EquivalenceVerdict(
                kind=VerdictKind.UNKNOWN,
                confidence=confidence,
                reason=f"Verified {verified}/{total} paths; "
                       f"{total - verified} paths inconclusive",
            )

        # No symbolic results — check fuzz results
        fuzz_iters = 0
        if self._state.fuzz_results:
            fuzz_iters = getattr(self._state.fuzz_results, 'iterations', 0)

        if fuzz_iters > 0 and not self._state.counterexamples:
            fuzz_conf = min(0.7, 0.3 + fuzz_iters / 10000)
            if structural_sim is not None and structural_sim >= 0.8:
                fuzz_conf = min(0.8, fuzz_conf + 0.15)
            return EquivalenceVerdict(
                kind=VerdictKind.UNKNOWN,
                confidence=fuzz_conf,
                reason=f"Fuzzing ({fuzz_iters} iterations) found no divergences"
                       + (f", structural similarity {structural_sim:.0%}"
                          if structural_sim is not None else "")
                       + "; symbolic verification incomplete",
            )

        # Structural similarity only
        if structural_sim is not None:
            if structural_sim >= 0.95:
                return EquivalenceVerdict(
                    kind=VerdictKind.UNKNOWN,
                    confidence=0.6,
                    reason=f"High structural similarity ({structural_sim:.0%}), "
                           f"but formal verification incomplete",
                )
            if structural_sim >= 0.5:
                return EquivalenceVerdict(
                    kind=VerdictKind.UNKNOWN,
                    confidence=structural_sim * 0.5,
                    reason=f"Moderate structural similarity ({structural_sim:.0%}), "
                           f"formal verification incomplete",
                )
            return EquivalenceVerdict(
                kind=VerdictKind.UNKNOWN,
                confidence=0.1,
                reason=f"Low structural similarity ({structural_sim:.0%}), "
                       f"likely divergent but not confirmed",
            )

        return EquivalenceVerdict(
            kind=VerdictKind.UNKNOWN,
            confidence=0.0,
            reason="Verification incomplete: insufficient analysis results",
        )


class PipelineBuilder:
    """Builder pattern for constructing customized pipelines."""

    def __init__(self):
        self._config = VerifyConfig.default()
        self._skip_phases: set = set()
        self._progress: Optional[ProgressCallback] = None

    def with_config(self, config: VerifyConfig) -> PipelineBuilder:
        self._config = config
        return self

    def with_profile(self, profile: str) -> PipelineBuilder:
        from .config import get_profile
        self._config = get_profile(profile)
        return self

    def skip_phase(self, phase: PipelinePhase) -> PipelineBuilder:
        self._skip_phases.add(phase)
        return self

    def with_progress(self, callback: ProgressCallback) -> PipelineBuilder:
        self._progress = callback
        return self

    def with_timeout(self, total: float) -> PipelineBuilder:
        self._config.timeouts.total_timeout = total
        return self

    def with_loop_bound(self, bound: int) -> PipelineBuilder:
        self._config.loops.default_bound = bound
        return self

    def build(self) -> VerificationPipeline:
        pipeline = VerificationPipeline(self._config)
        if self._progress:
            pipeline.set_progress_callback(self._progress)
        for phase in self._skip_phases:
            if phase in pipeline._phase_handlers:
                del pipeline._phase_handlers[phase]
        return pipeline
