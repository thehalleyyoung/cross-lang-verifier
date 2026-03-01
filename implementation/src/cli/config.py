"""Configuration for the Cross-Language Equivalence Verifier.

Provides VerifyConfig with all parameters, YAML loading, and profiles.
"""

from __future__ import annotations

import os
import json
import copy
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from ..utils.config_utils import (
    deep_merge, expand_env_recursive, resolve_path, validate_config,
    find_config_file,
)


class OutputFormat(Enum):
    """Output format for verification results."""
    JSON = "json"
    TEXT = "text"
    HTML = "html"
    SARIF = "sarif"


class Verbosity(Enum):
    """Verbosity level."""
    QUIET = 0
    NORMAL = 1
    VERBOSE = 2
    DEBUG = 3


class SMTSolverBackend(Enum):
    """SMT solver to use."""
    Z3 = "z3"
    CVC5 = "cvc5"
    BITWUZLA = "bitwuzla"


class FrontendMode(Enum):
    """Frontend parsing mode."""
    STRICT = "strict"
    LENIENT = "lenient"
    C2RUST = "c2rust"


class PathStrategy(Enum):
    """Symbolic execution path exploration strategy."""
    DFS = "dfs"
    BFS = "bfs"
    RANDOM = "random"
    COVERAGE_GUIDED = "coverage_guided"


@dataclass
class LoopConfig:
    """Loop handling configuration."""
    max_unroll: int = 10
    default_bound: int = 100
    allow_unbounded: bool = False
    widening_delay: int = 3

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> LoopConfig:
        return LoopConfig(**{k: v for k, v in d.items() if k in LoopConfig.__dataclass_fields__})


@dataclass
class TimeoutConfig:
    """Timeout configuration for various phases."""
    total_timeout: float = 300.0
    parse_timeout: float = 30.0
    analysis_timeout: float = 60.0
    smt_timeout: float = 120.0
    fuzz_timeout: float = 60.0
    per_path_timeout: float = 10.0

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> TimeoutConfig:
        return TimeoutConfig(**{k: v for k, v in d.items() if k in TimeoutConfig.__dataclass_fields__})


@dataclass
class FuzzerConfig:
    """Fuzzer configuration."""
    enabled: bool = True
    seed_count: int = 1000
    max_iterations: int = 10000
    mutation_rate: float = 0.1
    coverage_target: float = 0.8
    minimize_inputs: bool = True
    use_boundary_values: bool = True
    parallel_workers: int = 1

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> FuzzerConfig:
        return FuzzerConfig(**{k: v for k, v in d.items() if k in FuzzerConfig.__dataclass_fields__})


@dataclass
class FrontendConfig:
    """Frontend parsing configuration."""
    c_mode: FrontendMode = FrontendMode.C2RUST
    rust_mode: FrontendMode = FrontendMode.STRICT
    include_paths: List[str] = field(default_factory=list)
    defines: Dict[str, str] = field(default_factory=dict)
    target_triple: str = "x86_64-unknown-linux-gnu"
    pointer_width: int = 64
    endian: str = "little"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["c_mode"] = self.c_mode.value
        d["rust_mode"] = self.rust_mode.value
        return d

    @staticmethod
    def from_dict(d: dict) -> FrontendConfig:
        d = dict(d)
        if "c_mode" in d:
            d["c_mode"] = FrontendMode(d["c_mode"])
        if "rust_mode" in d:
            d["rust_mode"] = FrontendMode(d["rust_mode"])
        return FrontendConfig(**{k: v for k, v in d.items() if k in FrontendConfig.__dataclass_fields__})


@dataclass
class SymbolicExecConfig:
    """Symbolic execution configuration."""
    strategy: PathStrategy = PathStrategy.DFS
    max_paths: int = 1000
    max_depth: int = 100
    use_caching: bool = True
    concretize_arrays: bool = False
    array_size_bound: int = 256

    def to_dict(self) -> dict:
        d = asdict(self)
        d["strategy"] = self.strategy.value
        return d

    @staticmethod
    def from_dict(d: dict) -> SymbolicExecConfig:
        d = dict(d)
        if "strategy" in d:
            d["strategy"] = PathStrategy(d["strategy"])
        return SymbolicExecConfig(**{k: v for k, v in d.items() if k in SymbolicExecConfig.__dataclass_fields__})


@dataclass
class SMTConfig:
    """SMT solver configuration."""
    backend: SMTSolverBackend = SMTSolverBackend.Z3
    timeout: float = 120.0
    check_models: bool = False
    produce_proofs: bool = False
    incremental: bool = True
    random_seed: int = 42

    def to_dict(self) -> dict:
        d = asdict(self)
        d["backend"] = self.backend.value
        return d

    @staticmethod
    def from_dict(d: dict) -> SMTConfig:
        d = dict(d)
        if "backend" in d:
            d["backend"] = SMTSolverBackend(d["backend"])
        return SMTConfig(**{k: v for k, v in d.items() if k in SMTConfig.__dataclass_fields__})


@dataclass
class AnalysisConfig:
    """Analysis pass configuration."""
    run_alias_analysis: bool = True
    run_constant_propagation: bool = True
    run_dead_code_elimination: bool = False
    field_sensitive_alias: bool = True
    interprocedural: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> AnalysisConfig:
        return AnalysisConfig(**{k: v for k, v in d.items() if k in AnalysisConfig.__dataclass_fields__})


@dataclass
class VerifyConfig:
    """Top-level verification configuration."""
    # General
    output_format: OutputFormat = OutputFormat.JSON
    verbosity: Verbosity = Verbosity.NORMAL
    output_file: Optional[str] = None
    report_dir: Optional[str] = None

    # Function matching
    c_function: str = ""
    rust_function: str = ""
    match_by_name: bool = True

    # Sub-configs
    loops: LoopConfig = field(default_factory=LoopConfig)
    timeouts: TimeoutConfig = field(default_factory=TimeoutConfig)
    fuzzer: FuzzerConfig = field(default_factory=FuzzerConfig)
    frontend: FrontendConfig = field(default_factory=FrontendConfig)
    symbolic: SymbolicExecConfig = field(default_factory=SymbolicExecConfig)
    smt: SMTConfig = field(default_factory=SMTConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    def to_dict(self) -> dict:
        d: dict = {
            "output_format": self.output_format.value,
            "verbosity": self.verbosity.value,
            "output_file": self.output_file,
            "report_dir": self.report_dir,
            "c_function": self.c_function,
            "rust_function": self.rust_function,
            "match_by_name": self.match_by_name,
            "loops": self.loops.to_dict(),
            "timeouts": self.timeouts.to_dict(),
            "fuzzer": self.fuzzer.to_dict(),
            "frontend": self.frontend.to_dict(),
            "symbolic": self.symbolic.to_dict(),
            "smt": self.smt.to_dict(),
            "analysis": self.analysis.to_dict(),
        }
        return d

    @staticmethod
    def from_dict(d: dict) -> VerifyConfig:
        config = VerifyConfig()
        if "output_format" in d:
            config.output_format = OutputFormat(d["output_format"])
        if "verbosity" in d:
            config.verbosity = Verbosity(d["verbosity"])
        config.output_file = d.get("output_file")
        config.report_dir = d.get("report_dir")
        config.c_function = d.get("c_function", "")
        config.rust_function = d.get("rust_function", "")
        config.match_by_name = d.get("match_by_name", True)
        if "loops" in d:
            config.loops = LoopConfig.from_dict(d["loops"])
        if "timeouts" in d:
            config.timeouts = TimeoutConfig.from_dict(d["timeouts"])
        if "fuzzer" in d:
            config.fuzzer = FuzzerConfig.from_dict(d["fuzzer"])
        if "frontend" in d:
            config.frontend = FrontendConfig.from_dict(d["frontend"])
        if "symbolic" in d:
            config.symbolic = SymbolicExecConfig.from_dict(d["symbolic"])
        if "smt" in d:
            config.smt = SMTConfig.from_dict(d["smt"])
        if "analysis" in d:
            config.analysis = AnalysisConfig.from_dict(d["analysis"])
        return config

    def validate(self) -> List[str]:
        """Validate configuration. Returns list of errors."""
        errors: List[str] = []
        if self.timeouts.total_timeout <= 0:
            errors.append("total_timeout must be positive")
        if self.timeouts.smt_timeout <= 0:
            errors.append("smt_timeout must be positive")
        if self.loops.max_unroll < 0:
            errors.append("max_unroll must be non-negative")
        if self.fuzzer.seed_count <= 0:
            errors.append("seed_count must be positive")
        if not 0 <= self.fuzzer.mutation_rate <= 1:
            errors.append("mutation_rate must be in [0, 1]")
        if self.symbolic.max_paths <= 0:
            errors.append("max_paths must be positive")
        return errors

    def merge_with(self, override: Dict[str, Any]) -> VerifyConfig:
        """Create a new config by merging overrides into this config."""
        base = self.to_dict()
        merged = deep_merge(base, override)
        return VerifyConfig.from_dict(merged)

    @staticmethod
    def default() -> VerifyConfig:
        return VerifyConfig()

    @staticmethod
    def fast() -> VerifyConfig:
        """Fast profile: reduced bounds and timeouts for quick checking."""
        config = VerifyConfig()
        config.loops.max_unroll = 3
        config.loops.default_bound = 10
        config.timeouts.total_timeout = 30.0
        config.timeouts.smt_timeout = 10.0
        config.fuzzer.seed_count = 100
        config.fuzzer.max_iterations = 1000
        config.symbolic.max_paths = 100
        config.symbolic.max_depth = 20
        return config

    @staticmethod
    def thorough() -> VerifyConfig:
        """Thorough profile: higher bounds for comprehensive checking."""
        config = VerifyConfig()
        config.loops.max_unroll = 50
        config.loops.default_bound = 500
        config.timeouts.total_timeout = 600.0
        config.timeouts.smt_timeout = 300.0
        config.fuzzer.seed_count = 5000
        config.fuzzer.max_iterations = 100000
        config.symbolic.max_paths = 10000
        config.symbolic.max_depth = 500
        config.analysis.interprocedural = True
        return config

    @staticmethod
    def fuzz_only() -> VerifyConfig:
        """Fuzzing-only profile: skip symbolic execution and SMT."""
        config = VerifyConfig()
        config.fuzzer.enabled = True
        config.fuzzer.max_iterations = 50000
        config.symbolic.max_paths = 0
        config.timeouts.smt_timeout = 0
        return config

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @staticmethod
    def from_json(text: str) -> VerifyConfig:
        return VerifyConfig.from_dict(json.loads(text))

    @staticmethod
    def load_file(path: str) -> VerifyConfig:
        """Load config from a JSON or YAML file."""
        path = resolve_path(path)
        with open(path, "r") as f:
            text = f.read()

        if path.endswith((".yaml", ".yml")):
            try:
                import yaml
                data = yaml.safe_load(text)
            except ImportError:
                raise ImportError("PyYAML required for YAML config files")
        else:
            data = json.loads(text)

        data = expand_env_recursive(data)
        return VerifyConfig.from_dict(data)

    @staticmethod
    def find_and_load() -> VerifyConfig:
        """Search for a config file and load it, or return defaults."""
        config_names = [
            ".xlev.json", ".xlev.yaml", ".xlev.yml",
            "xlev.json", "xlev.yaml", "xlev.yml",
        ]
        path = find_config_file(config_names)
        if path:
            return VerifyConfig.load_file(path)
        return VerifyConfig.default()


# ---------------------------------------------------------------------------
# Profile registry
# ---------------------------------------------------------------------------

_PROFILES: Dict[str, VerifyConfig] = {
    "default": VerifyConfig.default(),
    "fast": VerifyConfig.fast(),
    "thorough": VerifyConfig.thorough(),
    "fuzz-only": VerifyConfig.fuzz_only(),
}


def get_profile(name: str) -> VerifyConfig:
    """Get a named config profile."""
    if name not in _PROFILES:
        available = ", ".join(_PROFILES.keys())
        raise ValueError(f"Unknown profile {name!r}. Available: {available}")
    return copy.deepcopy(_PROFILES[name])


def register_profile(name: str, config: VerifyConfig) -> None:
    """Register a custom config profile."""
    _PROFILES[name] = copy.deepcopy(config)


def list_profiles() -> List[str]:
    """Return list of available profile names."""
    return list(_PROFILES.keys())
