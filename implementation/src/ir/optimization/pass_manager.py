"""
Optimization pass infrastructure for the Cross-Language Equivalence Verifier.

Provides:
- Pass / FunctionPass / ModulePass base classes
- PassManager with ordering, dependencies, and iterative convergence
- AnalysisManager for caching analysis results
- Pass pipelines (O0, O1, O2)
- Pass statistics tracking
- Debug-mode verification after each pass
"""

from __future__ import annotations

import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type

from ...ir.function import Function
from ...ir.module import Module
from ...ir.basic_block import BasicBlock
from ...ir.validator import IRValidator

logger = logging.getLogger(__name__)


# ─── Pass Result ─────────────────────────────────────────────────────────

class PassResult(Enum):
    """Result of running a pass."""
    UNCHANGED = auto()
    CHANGED = auto()
    ERROR = auto()

    def __bool__(self) -> bool:
        return self == PassResult.CHANGED

    def merge(self, other: "PassResult") -> "PassResult":
        if self == PassResult.ERROR or other == PassResult.ERROR:
            return PassResult.ERROR
        if self == PassResult.CHANGED or other == PassResult.CHANGED:
            return PassResult.CHANGED
        return PassResult.UNCHANGED


# ─── Pass Statistics ────────────────────────────────────────────────────

@dataclass
class PassStatistics:
    """Statistics collected during pass execution."""
    pass_name: str
    num_runs: int = 0
    num_changes: int = 0
    total_time_ms: float = 0.0
    instructions_removed: int = 0
    instructions_added: int = 0
    blocks_removed: int = 0
    blocks_added: int = 0
    functions_modified: int = 0
    custom_counters: Dict[str, int] = field(default_factory=dict)

    def record_run(self, elapsed_ms: float, changed: bool) -> None:
        self.num_runs += 1
        self.total_time_ms += elapsed_ms
        if changed:
            self.num_changes += 1

    def increment(self, counter_name: str, amount: int = 1) -> None:
        self.custom_counters[counter_name] = self.custom_counters.get(counter_name, 0) + amount

    @property
    def average_time_ms(self) -> float:
        return self.total_time_ms / max(self.num_runs, 1)

    @property
    def change_rate(self) -> float:
        return self.num_changes / max(self.num_runs, 1)

    def summary(self) -> str:
        lines = [
            f"Pass: {self.pass_name}",
            f"  Runs: {self.num_runs}, Changes: {self.num_changes} ({self.change_rate:.1%})",
            f"  Time: {self.total_time_ms:.1f}ms total, {self.average_time_ms:.1f}ms avg",
        ]
        if self.instructions_removed or self.instructions_added:
            lines.append(f"  Instructions: +{self.instructions_added} -{self.instructions_removed}")
        if self.blocks_removed or self.blocks_added:
            lines.append(f"  Blocks: +{self.blocks_added} -{self.blocks_removed}")
        if self.functions_modified:
            lines.append(f"  Functions modified: {self.functions_modified}")
        for name, count in sorted(self.custom_counters.items()):
            lines.append(f"  {name}: {count}")
        return "\n".join(lines)

    def reset(self) -> None:
        self.num_runs = 0
        self.num_changes = 0
        self.total_time_ms = 0.0
        self.instructions_removed = 0
        self.instructions_added = 0
        self.blocks_removed = 0
        self.blocks_added = 0
        self.functions_modified = 0
        self.custom_counters.clear()


# ─── Pass Base Classes ──────────────────────────────────────────────────

class Pass(ABC):
    """Abstract base class for all optimization passes."""

    _name: str = ""
    _description: str = ""
    _required_analyses: List[str] = []
    _invalidated_analyses: List[str] = []
    _dependencies: List[str] = []

    def __init__(self) -> None:
        self._stats = PassStatistics(pass_name=self.name)
        self._enabled = True
        self._debug_verify = False

    @property
    def name(self) -> str:
        return self._name or self.__class__.__name__

    @property
    def description(self) -> str:
        return self._description or f"Optimization pass: {self.name}"

    @property
    def stats(self) -> PassStatistics:
        return self._stats

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    @property
    def debug_verify(self) -> bool:
        return self._debug_verify

    @debug_verify.setter
    def debug_verify(self, value: bool) -> None:
        self._debug_verify = value

    @property
    def required_analyses(self) -> List[str]:
        return self._required_analyses

    @property
    def invalidated_analyses(self) -> List[str]:
        return self._invalidated_analyses

    @property
    def dependencies(self) -> List[str]:
        return self._dependencies

    def initialize(self, **kwargs: Any) -> None:
        """Optional initialization hook called before first run."""
        pass

    def finalize(self) -> None:
        """Optional finalization hook called after last run."""
        pass


class FunctionPass(Pass):
    """A pass that operates on individual functions."""

    @abstractmethod
    def run_on_function(self, function: Function, analyses: "AnalysisManager") -> PassResult:
        """Run this pass on a single function. Return whether anything changed."""
        ...

    def should_skip_function(self, function: Function) -> bool:
        """Override to skip certain functions (e.g., declarations)."""
        if function.num_blocks == 0:
            return True
        return False


class ModulePass(Pass):
    """A pass that operates on an entire module."""

    @abstractmethod
    def run_on_module(self, module: Module, analyses: "AnalysisManager") -> PassResult:
        """Run this pass on an entire module. Return whether anything changed."""
        ...


# ─── Analysis Manager ──────────────────────────────────────────────────

class AnalysisResult:
    """Base class for analysis results."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._valid = True
        self._timestamp = time.monotonic()

    @property
    def name(self) -> str:
        return self._name

    @property
    def valid(self) -> bool:
        return self._valid

    def invalidate(self) -> None:
        self._valid = False

    @property
    def age_ms(self) -> float:
        return (time.monotonic() - self._timestamp) * 1000


class AnalysisProvider(ABC):
    """Interface for computing an analysis on demand."""

    @property
    @abstractmethod
    def analysis_name(self) -> str:
        ...

    @abstractmethod
    def compute_for_function(self, function: Function) -> Any:
        ...

    def compute_for_module(self, module: Module) -> Any:
        raise NotImplementedError(f"{self.analysis_name} does not support module-level analysis")


class AnalysisManager:
    """Manages analysis computation and caching.

    Analyses are computed lazily on first request and cached until invalidated.
    Passes declare which analyses they require and which they invalidate.
    """

    def __init__(self) -> None:
        self._providers: Dict[str, AnalysisProvider] = {}
        self._function_cache: Dict[Tuple[str, int], Any] = {}
        self._module_cache: Dict[str, Any] = {}
        self._computation_times: Dict[str, float] = {}
        self._hit_count: int = 0
        self._miss_count: int = 0

    def register_provider(self, provider: AnalysisProvider) -> None:
        self._providers[provider.analysis_name] = provider

    def register_lambda(self, name: str,
                        func_compute: Callable[[Function], Any]) -> None:
        class LambdaProvider(AnalysisProvider):
            @property
            def analysis_name(self) -> str:
                return name

            def compute_for_function(self, function: Function) -> Any:
                return func_compute(function)

        self._providers[name] = LambdaProvider()

    def get_function_analysis(self, name: str, function: Function) -> Any:
        key = (name, id(function))
        if key in self._function_cache:
            self._hit_count += 1
            return self._function_cache[key]

        self._miss_count += 1
        if name not in self._providers:
            raise KeyError(f"No provider registered for analysis '{name}'")

        start = time.monotonic()
        result = self._providers[name].compute_for_function(function)
        elapsed = (time.monotonic() - start) * 1000
        self._computation_times[name] = self._computation_times.get(name, 0.0) + elapsed

        self._function_cache[key] = result
        return result

    def get_module_analysis(self, name: str, module: Module) -> Any:
        if name in self._module_cache:
            self._hit_count += 1
            return self._module_cache[name]

        self._miss_count += 1
        if name not in self._providers:
            raise KeyError(f"No provider registered for analysis '{name}'")

        start = time.monotonic()
        result = self._providers[name].compute_for_module(module)
        elapsed = (time.monotonic() - start) * 1000
        self._computation_times[name] = self._computation_times.get(name, 0.0) + elapsed

        self._module_cache[name] = result
        return result

    def invalidate_function(self, name: str, function: Function) -> None:
        key = (name, id(function))
        self._function_cache.pop(key, None)

    def invalidate_all_for_function(self, function: Function) -> None:
        keys_to_remove = [k for k in self._function_cache if k[1] == id(function)]
        for key in keys_to_remove:
            del self._function_cache[key]

    def invalidate_module(self, name: str) -> None:
        self._module_cache.pop(name, None)

    def invalidate_analyses(self, names: List[str], function: Optional[Function] = None) -> None:
        for name in names:
            if function is not None:
                self.invalidate_function(name, function)
            else:
                self.invalidate_module(name)
                func_keys = [k for k in self._function_cache if k[0] == name]
                for key in func_keys:
                    del self._function_cache[key]

    def invalidate_all(self) -> None:
        self._function_cache.clear()
        self._module_cache.clear()

    @property
    def cache_hit_rate(self) -> float:
        total = self._hit_count + self._miss_count
        return self._hit_count / max(total, 1)

    def statistics(self) -> str:
        total = self._hit_count + self._miss_count
        lines = [
            f"AnalysisManager Statistics:",
            f"  Cache hits: {self._hit_count}/{total} ({self.cache_hit_rate:.1%})",
            f"  Function cache entries: {len(self._function_cache)}",
            f"  Module cache entries: {len(self._module_cache)}",
        ]
        if self._computation_times:
            lines.append("  Computation times:")
            for name, ms in sorted(self._computation_times.items()):
                lines.append(f"    {name}: {ms:.1f}ms")
        return "\n".join(lines)


# ─── Pass Registry ──────────────────────────────────────────────────────

class PassRegistry:
    """Global registry of available passes."""

    _instance: Optional["PassRegistry"] = None

    def __init__(self) -> None:
        self._passes: Dict[str, Type[Pass]] = {}
        self._descriptions: Dict[str, str] = {}
        self._categories: Dict[str, List[str]] = {}

    @classmethod
    def instance(cls) -> "PassRegistry":
        if cls._instance is None:
            cls._instance = PassRegistry()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    def register(self, pass_class: Type[Pass], category: str = "general") -> None:
        name = pass_class._name or pass_class.__name__
        self._passes[name] = pass_class
        self._descriptions[name] = pass_class._description or ""
        if category not in self._categories:
            self._categories[category] = []
        self._categories[category].append(name)

    def get(self, name: str) -> Optional[Type[Pass]]:
        return self._passes.get(name)

    def create(self, name: str) -> Pass:
        pass_class = self._passes.get(name)
        if pass_class is None:
            raise KeyError(f"Unknown pass: {name}")
        return pass_class()

    def available_passes(self) -> List[str]:
        return sorted(self._passes.keys())

    def passes_in_category(self, category: str) -> List[str]:
        return self._categories.get(category, [])

    def categories(self) -> List[str]:
        return sorted(self._categories.keys())

    def describe(self, name: str) -> str:
        return self._descriptions.get(name, "No description available")

    def list_all(self) -> str:
        lines = ["Available passes:"]
        for cat in sorted(self._categories.keys()):
            lines.append(f"\n  [{cat}]")
            for name in sorted(self._categories[cat]):
                desc = self._descriptions.get(name, "")
                lines.append(f"    {name}: {desc}")
        return "\n".join(lines)


def register_pass(category: str = "general") -> Callable:
    """Decorator for registering a pass with the global registry."""
    def decorator(cls: Type[Pass]) -> Type[Pass]:
        PassRegistry.instance().register(cls, category)
        return cls
    return decorator


# ─── Pass Manager ───────────────────────────────────────────────────────

@dataclass
class PassManagerConfig:
    """Configuration for the PassManager."""
    max_iterations: int = 10
    debug_verify: bool = False
    collect_statistics: bool = True
    verbose: bool = False
    timeout_per_function_ms: float = 30000.0
    bail_on_error: bool = True
    fixed_point: bool = False


class PassManager:
    """Manages and executes a sequence of optimization passes.

    Features:
    - Ordered pass execution with dependency resolution
    - Iterative convergence (fixed-point) mode
    - Analysis caching and invalidation
    - Debug verification after each pass
    - Statistics collection and reporting
    """

    def __init__(self, config: Optional[PassManagerConfig] = None) -> None:
        self._config = config or PassManagerConfig()
        self._passes: List[Pass] = []
        self._analyses = AnalysisManager()
        self._all_stats: List[PassStatistics] = []
        self._total_time_ms: float = 0.0
        self._run_count: int = 0
        self._validator = IRValidator()

    @property
    def config(self) -> PassManagerConfig:
        return self._config

    @property
    def analyses(self) -> AnalysisManager:
        return self._analyses

    def add_pass(self, pass_instance: Pass) -> "PassManager":
        if self._config.debug_verify:
            pass_instance.debug_verify = True
        self._passes.append(pass_instance)
        self._all_stats.append(pass_instance.stats)
        return self

    def add_passes(self, passes: List[Pass]) -> "PassManager":
        for p in passes:
            self.add_pass(p)
        return self

    def insert_pass_before(self, before_name: str, pass_instance: Pass) -> "PassManager":
        for i, p in enumerate(self._passes):
            if p.name == before_name:
                if self._config.debug_verify:
                    pass_instance.debug_verify = True
                self._passes.insert(i, pass_instance)
                self._all_stats.append(pass_instance.stats)
                return self
        raise KeyError(f"Pass '{before_name}' not found")

    def insert_pass_after(self, after_name: str, pass_instance: Pass) -> "PassManager":
        for i, p in enumerate(self._passes):
            if p.name == after_name:
                if self._config.debug_verify:
                    pass_instance.debug_verify = True
                self._passes.insert(i + 1, pass_instance)
                self._all_stats.append(pass_instance.stats)
                return self
        raise KeyError(f"Pass '{after_name}' not found")

    def remove_pass(self, name: str) -> "PassManager":
        self._passes = [p for p in self._passes if p.name != name]
        return self

    def clear_passes(self) -> "PassManager":
        self._passes.clear()
        self._all_stats.clear()
        return self

    def _resolve_dependencies(self) -> List[Pass]:
        """Topologically sort passes based on dependencies."""
        name_to_pass = {p.name: p for p in self._passes}
        visited: Set[str] = set()
        result: List[Pass] = []
        temp_mark: Set[str] = set()

        def visit(name: str) -> None:
            if name in temp_mark:
                raise ValueError(f"Circular dependency involving pass '{name}'")
            if name in visited:
                return
            temp_mark.add(name)
            p = name_to_pass.get(name)
            if p:
                for dep in p.dependencies:
                    if dep in name_to_pass:
                        visit(dep)
            temp_mark.discard(name)
            visited.add(name)
            if p:
                result.append(p)

        for p in self._passes:
            visit(p.name)

        return result

    def _verify_function(self, function: Function, pass_name: str) -> None:
        errors = function.validate()
        if errors:
            msg = f"Verification failed after pass '{pass_name}': " + "; ".join(errors[:5])
            logger.error(msg)
            if self._config.bail_on_error:
                raise RuntimeError(msg)

    def _run_function_pass(self, fpass: FunctionPass, function: Function) -> PassResult:
        if fpass.should_skip_function(function):
            return PassResult.UNCHANGED

        for analysis_name in fpass.required_analyses:
            try:
                self._analyses.get_function_analysis(analysis_name, function)
            except KeyError:
                logger.warning(f"Pass {fpass.name} requires unavailable analysis '{analysis_name}'")

        start = time.monotonic()
        try:
            result = fpass.run_on_function(function, self._analyses)
        except Exception as e:
            logger.error(f"Pass {fpass.name} failed on {function.name}: {e}")
            fpass.stats.record_run(0, False)
            return PassResult.ERROR
        elapsed = (time.monotonic() - start) * 1000
        fpass.stats.record_run(elapsed, result == PassResult.CHANGED)

        if result == PassResult.CHANGED:
            fpass.stats.functions_modified += 1
            self._analyses.invalidate_analyses(fpass.invalidated_analyses, function)

        if fpass.debug_verify and result == PassResult.CHANGED:
            self._verify_function(function, fpass.name)

        return result

    def run_on_function(self, function: Function) -> PassResult:
        """Run all passes on a single function."""
        overall = PassResult.UNCHANGED
        sorted_passes = self._resolve_dependencies()

        if self._config.fixed_point:
            return self._run_fixed_point_function(sorted_passes, function)

        for p in sorted_passes:
            if not p.enabled:
                continue
            if isinstance(p, FunctionPass):
                result = self._run_function_pass(p, function)
                overall = overall.merge(result)
                if result == PassResult.ERROR and self._config.bail_on_error:
                    return PassResult.ERROR

        return overall

    def _run_fixed_point_function(self, passes: List[Pass], function: Function) -> PassResult:
        """Run passes iteratively until no more changes occur."""
        overall = PassResult.UNCHANGED
        for iteration in range(self._config.max_iterations):
            changed_this_iter = False
            for p in passes:
                if not p.enabled or not isinstance(p, FunctionPass):
                    continue
                result = self._run_function_pass(p, function)
                if result == PassResult.ERROR and self._config.bail_on_error:
                    return PassResult.ERROR
                if result == PassResult.CHANGED:
                    changed_this_iter = True
                    overall = PassResult.CHANGED
            if not changed_this_iter:
                logger.debug(f"Fixed point reached after {iteration + 1} iterations")
                break
        else:
            logger.warning(f"Fixed point not reached after {self._config.max_iterations} iterations")
        return overall

    def run_on_module(self, module: Module) -> PassResult:
        """Run all passes on an entire module."""
        overall = PassResult.UNCHANGED
        sorted_passes = self._resolve_dependencies()
        start = time.monotonic()
        self._run_count += 1

        for p in sorted_passes:
            if not p.enabled:
                continue

            if isinstance(p, ModulePass):
                s = time.monotonic()
                try:
                    result = p.run_on_module(module, self._analyses)
                except Exception as e:
                    logger.error(f"Module pass {p.name} failed: {e}")
                    p.stats.record_run(0, False)
                    if self._config.bail_on_error:
                        return PassResult.ERROR
                    continue
                elapsed = (time.monotonic() - s) * 1000
                p.stats.record_run(elapsed, result == PassResult.CHANGED)
                if result == PassResult.CHANGED:
                    self._analyses.invalidate_analyses(p.invalidated_analyses)
                overall = overall.merge(result)

            elif isinstance(p, FunctionPass):
                for func in module.functions:
                    result = self._run_function_pass(p, func)
                    overall = overall.merge(result)
                    if result == PassResult.ERROR and self._config.bail_on_error:
                        self._total_time_ms += (time.monotonic() - start) * 1000
                        return PassResult.ERROR

        self._total_time_ms += (time.monotonic() - start) * 1000

        if self._config.fixed_point:
            return self._run_fixed_point_module(sorted_passes, module)

        return overall

    def _run_fixed_point_module(self, passes: List[Pass], module: Module) -> PassResult:
        overall = PassResult.UNCHANGED
        for iteration in range(self._config.max_iterations):
            changed = False
            for p in passes:
                if not p.enabled:
                    continue
                if isinstance(p, ModulePass):
                    result = p.run_on_module(module, self._analyses)
                    if result == PassResult.CHANGED:
                        changed = True
                        self._analyses.invalidate_analyses(p.invalidated_analyses)
                elif isinstance(p, FunctionPass):
                    for func in module.functions:
                        result = self._run_function_pass(p, func)
                        if result == PassResult.CHANGED:
                            changed = True
            if changed:
                overall = PassResult.CHANGED
            else:
                break
        return overall

    def get_pass(self, name: str) -> Optional[Pass]:
        for p in self._passes:
            if p.name == name:
                return p
        return None

    def pass_names(self) -> List[str]:
        return [p.name for p in self._passes]

    def statistics_report(self) -> str:
        lines = [
            f"PassManager Statistics (run #{self._run_count})",
            f"Total time: {self._total_time_ms:.1f}ms",
            f"Passes: {len(self._passes)}",
            "",
        ]
        for s in self._all_stats:
            if s.num_runs > 0:
                lines.append(s.summary())
                lines.append("")
        lines.append(self._analyses.statistics())
        return "\n".join(lines)


# ─── Pass Pipeline ──────────────────────────────────────────────────────

class PassPipeline:
    """A named, reusable sequence of passes."""

    def __init__(self, name: str, description: str = "") -> None:
        self._name = name
        self._description = description
        self._pass_constructors: List[Callable[[], Pass]] = []
        self._sub_pipelines: List["PassPipeline"] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def add(self, constructor: Callable[[], Pass]) -> "PassPipeline":
        self._pass_constructors.append(constructor)
        return self

    def add_sub_pipeline(self, pipeline: "PassPipeline") -> "PassPipeline":
        self._sub_pipelines.append(pipeline)
        return self

    def build_passes(self) -> List[Pass]:
        result: List[Pass] = []
        for constructor in self._pass_constructors:
            result.append(constructor())
        for sub in self._sub_pipelines:
            result.extend(sub.build_passes())
        return result

    def populate_manager(self, manager: PassManager) -> PassManager:
        for p in self.build_passes():
            manager.add_pass(p)
        return manager

    def create_manager(self, config: Optional[PassManagerConfig] = None) -> PassManager:
        manager = PassManager(config)
        return self.populate_manager(manager)

    def __len__(self) -> int:
        count = len(self._pass_constructors)
        for sub in self._sub_pipelines:
            count += len(sub)
        return count

    def __str__(self) -> str:
        names = []
        for c in self._pass_constructors:
            try:
                p = c()
                names.append(p.name)
            except Exception:
                names.append("<unknown>")
        for sub in self._sub_pipelines:
            names.append(f"[{sub.name}]")
        return f"Pipeline '{self._name}': {' -> '.join(names)}"


# ─── Standard Pipelines ────────────────────────────────────────────────

def _import_pass_classes() -> Dict[str, Type[Pass]]:
    """Lazily import pass classes to avoid circular imports."""
    classes: Dict[str, Type[Pass]] = {}
    try:
        from .dce import DeadCodeElimination, DeadBlockElimination
        classes["dce"] = DeadCodeElimination
        classes["dead_block_elim"] = DeadBlockElimination
    except ImportError:
        pass
    try:
        from .constant_fold import ConstantFolder, ConstantPropagation
        classes["constant_fold"] = ConstantFolder
        classes["constant_prop"] = ConstantPropagation
    except ImportError:
        pass
    try:
        from .mem2reg import Mem2Reg
        classes["mem2reg"] = Mem2Reg
    except ImportError:
        pass
    try:
        from .simplify import InstructionSimplifier
        classes["simplify"] = InstructionSimplifier
    except ImportError:
        pass
    try:
        from .inline import FunctionInliner
        classes["inline"] = FunctionInliner
    except ImportError:
        pass
    try:
        from .gvn import GlobalValueNumbering
        classes["gvn"] = GlobalValueNumbering
    except ImportError:
        pass
    try:
        from .licm import LoopInvariantCodeMotion
        classes["licm"] = LoopInvariantCodeMotion
    except ImportError:
        pass
    try:
        from .sccp import SparseConditionalConstantPropagation
        classes["sccp"] = SparseConditionalConstantPropagation
    except ImportError:
        pass
    return classes


def create_pipeline_O0() -> PassPipeline:
    """No optimization – only structural cleanup."""
    pipeline = PassPipeline("O0", "No optimization, structural cleanup only")
    classes = _import_pass_classes()
    if "dead_block_elim" in classes:
        pipeline.add(classes["dead_block_elim"])
    return pipeline


def create_pipeline_O1() -> PassPipeline:
    """Basic optimizations: mem2reg, constant fold, DCE, simplify."""
    pipeline = PassPipeline("O1", "Basic optimizations")
    classes = _import_pass_classes()
    order = ["mem2reg", "constant_fold", "simplify", "dce", "dead_block_elim"]
    for name in order:
        if name in classes:
            pipeline.add(classes[name])
    return pipeline


def create_pipeline_O2() -> PassPipeline:
    """Aggressive optimizations: O1 + GVN, LICM, SCCP, inlining."""
    pipeline = PassPipeline("O2", "Aggressive optimizations")
    classes = _import_pass_classes()
    order = [
        "mem2reg", "sccp", "constant_fold", "simplify",
        "inline", "mem2reg", "gvn", "licm",
        "constant_fold", "simplify", "dce", "dead_block_elim",
    ]
    for name in order:
        if name in classes:
            pipeline.add(classes[name])
    return pipeline


# ─── Utility: Verification Pass ────────────────────────────────────────

class VerificationPass(FunctionPass):
    """Runs the IR validator as a pass. Useful in debug pipelines."""

    _name = "verify"
    _description = "Run IR validator to check structural integrity"

    def __init__(self) -> None:
        super().__init__()
        self._validator = IRValidator()
        self._errors_found: List[Tuple[str, List[str]]] = []

    def run_on_function(self, function: Function, analyses: AnalysisManager) -> PassResult:
        errors = function.validate()
        if errors:
            self._errors_found.append((function.name, errors))
            logger.error(f"Verification errors in {function.name}: {errors}")
            return PassResult.ERROR
        return PassResult.UNCHANGED

    @property
    def errors(self) -> List[Tuple[str, List[str]]]:
        return self._errors_found


class PrintPass(FunctionPass):
    """Debug pass that prints the IR of each function."""

    _name = "print_ir"
    _description = "Print IR for debugging"

    def __init__(self, prefix: str = "") -> None:
        super().__init__()
        self._prefix = prefix

    def run_on_function(self, function: Function, analyses: AnalysisManager) -> PassResult:
        header = f"=== {self._prefix}{function.name} ===" if self._prefix else f"=== {function.name} ==="
        logger.info(header)
        logger.info(function.dump())
        return PassResult.UNCHANGED


# ─── Pass Scheduling Utilities ──────────────────────────────────────────

class PassScheduler:
    """Schedules passes for optimal execution considering dependencies and invalidation."""

    def __init__(self) -> None:
        self._constraints: List[Tuple[str, str]] = []

    def add_ordering(self, before: str, after: str) -> None:
        self._constraints.append((before, after))

    def schedule(self, passes: List[Pass]) -> List[Pass]:
        name_to_pass = {p.name: p for p in passes}
        name_to_idx = {p.name: i for i, p in enumerate(passes)}

        all_constraints = list(self._constraints)
        for p in passes:
            for dep in p.dependencies:
                if dep in name_to_idx:
                    all_constraints.append((dep, p.name))

        # Kahn's algorithm
        in_degree: Dict[str, int] = {p.name: 0 for p in passes}
        adj: Dict[str, List[str]] = {p.name: [] for p in passes}
        for before, after in all_constraints:
            if before in adj and after in in_degree:
                adj[before].append(after)
                in_degree[after] += 1

        queue = sorted([n for n, d in in_degree.items() if d == 0])
        result: List[Pass] = []
        while queue:
            node = queue.pop(0)
            result.append(name_to_pass[node])
            for neighbor in sorted(adj.get(node, [])):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(passes):
            logger.warning("Cycle detected in pass scheduling; using original order")
            return passes
        return result


class FixedPointRunner:
    """Runs a set of passes until a fixed point is reached."""

    def __init__(self, passes: List[FunctionPass], max_iterations: int = 10) -> None:
        self._passes = passes
        self._max_iterations = max_iterations
        self._iterations_run = 0

    @property
    def iterations_run(self) -> int:
        return self._iterations_run

    def run(self, function: Function, analyses: AnalysisManager) -> PassResult:
        overall = PassResult.UNCHANGED
        for i in range(self._max_iterations):
            self._iterations_run = i + 1
            changed = False
            for p in self._passes:
                if not p.enabled:
                    continue
                if p.should_skip_function(function):
                    continue
                result = p.run_on_function(function, analyses)
                if result == PassResult.CHANGED:
                    changed = True
                    overall = PassResult.CHANGED
                    analyses.invalidate_analyses(p.invalidated_analyses, function)
            if not changed:
                break
        return overall


# ─── Interleaved Pass Execution ─────────────────────────────────────────

class InterleavedPassManager:
    """Runs passes in an interleaved fashion across functions.

    Instead of running all passes on func1, then all on func2, etc.,
    this runs pass1 on all functions, then pass2 on all functions, etc.
    This can be better for inter-procedural analyses.
    """

    def __init__(self, config: Optional[PassManagerConfig] = None) -> None:
        self._config = config or PassManagerConfig()
        self._passes: List[FunctionPass] = []
        self._analyses = AnalysisManager()

    def add_pass(self, p: FunctionPass) -> None:
        self._passes.append(p)

    def run_on_module(self, module: Module) -> PassResult:
        overall = PassResult.UNCHANGED
        functions = list(module.functions)

        for p in self._passes:
            if not p.enabled:
                continue
            for func in functions:
                if p.should_skip_function(func):
                    continue
                start = time.monotonic()
                result = p.run_on_function(func, self._analyses)
                elapsed = (time.monotonic() - start) * 1000
                p.stats.record_run(elapsed, result == PassResult.CHANGED)
                if result == PassResult.CHANGED:
                    overall = PassResult.CHANGED
                    self._analyses.invalidate_analyses(p.invalidated_analyses, func)

        return overall


# ─── Analysis Providers for Common Analyses ─────────────────────────────

class CFGAnalysisProvider(AnalysisProvider):
    """Provides CFG analysis on demand."""

    @property
    def analysis_name(self) -> str:
        return "cfg"

    def compute_for_function(self, function: Function) -> Any:
        from ...analysis.cfg import CFG
        return CFG(function)


class DominatorTreeProvider(AnalysisProvider):
    """Provides dominator tree analysis on demand."""

    @property
    def analysis_name(self) -> str:
        return "domtree"

    def compute_for_function(self, function: Function) -> Any:
        from ...analysis.cfg import CFG, DominatorTree
        cfg = CFG(function)
        return DominatorTree.build(cfg)


class LoopInfoProvider(AnalysisProvider):
    """Provides loop analysis on demand."""

    @property
    def analysis_name(self) -> str:
        return "loops"

    def compute_for_function(self, function: Function) -> Any:
        from ...analysis.cfg import CFG, LoopInfo
        cfg = CFG(function)
        return LoopInfo.build(cfg)


class AliasAnalysisProvider(AnalysisProvider):
    """Provides alias analysis on demand."""

    @property
    def analysis_name(self) -> str:
        return "alias"

    def compute_for_function(self, function: Function) -> Any:
        from ...analysis.alias import AndersenAnalysis
        aa = AndersenAnalysis()
        aa.analyze_function(function)
        return aa


def create_default_analysis_manager() -> AnalysisManager:
    """Create an AnalysisManager with all standard analysis providers registered."""
    am = AnalysisManager()
    am.register_provider(CFGAnalysisProvider())
    am.register_provider(DominatorTreeProvider())
    am.register_provider(LoopInfoProvider())
    am.register_provider(AliasAnalysisProvider())
    return am
