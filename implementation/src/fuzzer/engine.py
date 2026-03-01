"""
Fuzzing engine for differential testing.

Manages fuzzing campaigns with random/mutation-based/grammar-based input
generation, coverage tracking, corpus management, and counterexample
detection.
"""

from __future__ import annotations

import hashlib
import os
import random
import struct
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    List, Optional, Dict, Tuple, Any, Set, Callable, Sequence,
)

from ..ir.types import IRType, IntType, FloatType, PointerType, VoidType, Signedness, FloatKind
from ..ir.function import Function


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class MutationStrategy(Enum):
    BIT_FLIP = auto()
    BYTE_FLIP = auto()
    ARITHMETIC = auto()
    INTERESTING_VALUE = auto()
    BLOCK_DELETION = auto()
    BLOCK_INSERTION = auto()
    BLOCK_OVERWRITE = auto()
    DICTIONARY = auto()
    CROSSOVER = auto()
    RANDOM = auto()


@dataclass
class FuzzInput:
    """A single fuzzing input."""
    data: bytes
    typed_values: Dict[str, Any] = field(default_factory=dict)
    source: str = "random"
    generation: int = 0
    energy: float = 1.0
    coverage_hash: Optional[str] = None

    @property
    def size(self) -> int:
        return len(self.data)

    def get_int(self, name: str, width: int = 32, signed: bool = True) -> int:
        val = self.typed_values.get(name)
        if val is not None:
            return int(val)
        # Extract from raw data
        if width <= 8:
            fmt = 'b' if signed else 'B'
        elif width <= 16:
            fmt = '<h' if signed else '<H'
        elif width <= 32:
            fmt = '<i' if signed else '<I'
        else:
            fmt = '<q' if signed else '<Q'
        size = struct.calcsize(fmt)
        if len(self.data) >= size:
            return struct.unpack(fmt, self.data[:size])[0]
        return 0

    def get_float(self, name: str, width: int = 64) -> float:
        val = self.typed_values.get(name)
        if val is not None:
            return float(val)
        fmt = '<f' if width <= 32 else '<d'
        size = struct.calcsize(fmt)
        if len(self.data) >= size:
            return struct.unpack(fmt, self.data[:size])[0]
        return 0.0

    def fingerprint(self) -> str:
        return hashlib.sha256(self.data).hexdigest()[:16]

    def __repr__(self) -> str:
        return f"FuzzInput({self.size}B, src={self.source}, gen={self.generation})"


@dataclass
class FuzzResult:
    """Result of executing a single fuzz input."""
    input: FuzzInput
    c_output: Optional[Any] = None
    rust_output: Optional[Any] = None
    c_crashed: bool = False
    rust_crashed: bool = False
    c_error: Optional[str] = None
    rust_error: Optional[str] = None
    is_divergent: bool = False
    divergence_description: str = ""
    execution_time_ms: float = 0.0
    new_coverage: bool = False

    @property
    def is_crash(self) -> bool:
        return self.c_crashed or self.rust_crashed

    @property
    def is_interesting(self) -> bool:
        return self.is_divergent or self.is_crash or self.new_coverage

    def __repr__(self) -> str:
        status = "DIVERGENT" if self.is_divergent else ("CRASH" if self.is_crash else "OK")
        return f"FuzzResult({status}, {self.execution_time_ms:.1f}ms)"


@dataclass
class FuzzConfig:
    """Configuration for the fuzzing engine."""
    max_iterations: int = 100000
    max_time_seconds: float = 3600.0
    max_input_size: int = 4096
    min_input_size: int = 1
    mutation_depth: int = 5
    dictionary: List[bytes] = field(default_factory=list)
    seed_inputs: List[FuzzInput] = field(default_factory=list)
    coverage_guided: bool = True
    save_corpus: bool = True
    corpus_dir: str = ""
    random_seed: int = 42


@dataclass
class CampaignStats:
    """Statistics for a fuzzing campaign."""
    total_inputs: int = 0
    total_time_ms: float = 0.0
    divergences_found: int = 0
    crashes_found: int = 0
    corpus_size: int = 0
    coverage_edges: int = 0
    executions_per_second: float = 0.0
    unique_crashes: int = 0
    unique_divergences: int = 0


# ---------------------------------------------------------------------------
# Input generation
# ---------------------------------------------------------------------------

class InputGenerator:
    """Generates fuzz inputs using various strategies."""

    INTERESTING_INTS_8 = [0, 1, -1, 127, -128, 255]
    INTERESTING_INTS_16 = [0, 1, -1, 255, 256, 32767, -32768, 65535]
    INTERESTING_INTS_32 = [
        0, 1, -1, 127, -128, 255, 256, 32767, -32768, 65535,
        65536, 2147483647, -2147483648, 4294967295,
    ]
    INTERESTING_INTS_64 = [
        0, 1, -1, 2147483647, -2147483648, 4294967295,
        9223372036854775807, -9223372036854775808,
    ]

    def __init__(self, config: FuzzConfig):
        self.config = config
        self.rng = random.Random(config.random_seed)

    def generate_random(self, size: Optional[int] = None) -> FuzzInput:
        """Generate a completely random input."""
        if size is None:
            size = self.rng.randint(self.config.min_input_size, self.config.max_input_size)
        data = bytes(self.rng.getrandbits(8) for _ in range(size))
        return FuzzInput(data=data, source="random")

    def generate_typed(self, param_types: List[Tuple[str, IRType]]) -> FuzzInput:
        """Generate a type-aware random input."""
        typed_values: Dict[str, Any] = {}
        data_parts: List[bytes] = []

        for name, ty in param_types:
            if isinstance(ty, IntType):
                val = self.rng.randint(0, (1 << ty.width) - 1)
                if ty.signed and val >= (1 << (ty.width - 1)):
                    val -= (1 << ty.width)
                typed_values[name] = val
                byte_count = max(ty.width // 8, 1)
                data_parts.append(val.to_bytes(byte_count, 'little', signed=ty.signed))
            elif isinstance(ty, FloatType):
                val = self.rng.uniform(-1e10, 1e10)
                typed_values[name] = val
                if ty.kind == FloatKind.F32:
                    data_parts.append(struct.pack('<f', val))
                else:
                    data_parts.append(struct.pack('<d', val))
            elif isinstance(ty, PointerType):
                typed_values[name] = 0
                data_parts.append(b'\x00' * 8)
            else:
                typed_values[name] = 0
                data_parts.append(b'\x00\x00\x00\x00')

        data = b''.join(data_parts)
        return FuzzInput(data=data, typed_values=typed_values, source="typed_random")

    def mutate(self, inp: FuzzInput) -> FuzzInput:
        """Mutate an input using a random strategy."""
        strategy = self.rng.choice(list(MutationStrategy))
        data = bytearray(inp.data)

        if strategy == MutationStrategy.BIT_FLIP:
            data = self._bit_flip(data)
        elif strategy == MutationStrategy.BYTE_FLIP:
            data = self._byte_flip(data)
        elif strategy == MutationStrategy.ARITHMETIC:
            data = self._arithmetic(data)
        elif strategy == MutationStrategy.INTERESTING_VALUE:
            data = self._interesting_value(data)
        elif strategy == MutationStrategy.BLOCK_DELETION:
            data = self._block_delete(data)
        elif strategy == MutationStrategy.BLOCK_INSERTION:
            data = self._block_insert(data)
        elif strategy == MutationStrategy.BLOCK_OVERWRITE:
            data = self._block_overwrite(data)
        elif strategy == MutationStrategy.RANDOM:
            data = self._random_bytes(data)
        else:
            data = self._bit_flip(data)

        return FuzzInput(
            data=bytes(data),
            source=f"mutate_{strategy.name}",
            generation=inp.generation + 1,
            energy=inp.energy * 0.9,
        )

    def crossover(self, a: FuzzInput, b: FuzzInput) -> FuzzInput:
        """Cross two inputs."""
        if len(a.data) == 0 or len(b.data) == 0:
            return a

        point = self.rng.randint(0, min(len(a.data), len(b.data)))
        data = a.data[:point] + b.data[point:]
        return FuzzInput(data=data, source="crossover", generation=max(a.generation, b.generation) + 1)

    # -- Mutation strategies --

    def _bit_flip(self, data: bytearray) -> bytearray:
        if not data:
            return data
        pos = self.rng.randint(0, len(data) - 1)
        bit = self.rng.randint(0, 7)
        data[pos] ^= (1 << bit)
        return data

    def _byte_flip(self, data: bytearray) -> bytearray:
        if not data:
            return data
        pos = self.rng.randint(0, len(data) - 1)
        data[pos] ^= 0xFF
        return data

    def _arithmetic(self, data: bytearray) -> bytearray:
        if len(data) < 4:
            return self._bit_flip(data)
        pos = self.rng.randint(0, len(data) - 4)
        val = int.from_bytes(data[pos:pos + 4], 'little', signed=True)
        delta = self.rng.randint(-35, 35)
        val = (val + delta) & 0xFFFFFFFF
        data[pos:pos + 4] = val.to_bytes(4, 'little')
        return data

    def _interesting_value(self, data: bytearray) -> bytearray:
        if len(data) < 4:
            return self._bit_flip(data)
        pos = self.rng.randint(0, len(data) - 4)
        val = self.rng.choice(self.INTERESTING_INTS_32)
        data[pos:pos + 4] = (val & 0xFFFFFFFF).to_bytes(4, 'little')
        return data

    def _block_delete(self, data: bytearray) -> bytearray:
        if len(data) <= 1:
            return data
        start = self.rng.randint(0, len(data) - 1)
        length = self.rng.randint(1, min(32, len(data) - start))
        del data[start:start + length]
        return data

    def _block_insert(self, data: bytearray) -> bytearray:
        pos = self.rng.randint(0, len(data))
        length = self.rng.randint(1, 32)
        insert_data = bytes(self.rng.getrandbits(8) for _ in range(length))
        data[pos:pos] = insert_data
        # Trim to max size
        if len(data) > self.config.max_input_size:
            data = data[:self.config.max_input_size]
        return data

    def _block_overwrite(self, data: bytearray) -> bytearray:
        if not data:
            return data
        pos = self.rng.randint(0, len(data) - 1)
        length = self.rng.randint(1, min(32, len(data) - pos))
        for i in range(length):
            data[pos + i] = self.rng.getrandbits(8)
        return data

    def _random_bytes(self, data: bytearray) -> bytearray:
        if not data:
            return data
        num = self.rng.randint(1, min(8, len(data)))
        for _ in range(num):
            pos = self.rng.randint(0, len(data) - 1)
            data[pos] = self.rng.getrandbits(8)
        return data


# ---------------------------------------------------------------------------
# Corpus management
# ---------------------------------------------------------------------------

class Corpus:
    """Manages a corpus of interesting inputs."""

    def __init__(self):
        self._inputs: List[FuzzInput] = []
        self._fingerprints: Set[str] = set()
        self._coverage_hashes: Set[str] = set()

    def add(self, inp: FuzzInput) -> bool:
        """Add an input to the corpus. Returns True if new."""
        fp = inp.fingerprint()
        if fp in self._fingerprints:
            return False
        self._fingerprints.add(fp)
        self._inputs.append(inp)
        if inp.coverage_hash:
            self._coverage_hashes.add(inp.coverage_hash)
        return True

    def pick_random(self, rng: random.Random) -> Optional[FuzzInput]:
        if not self._inputs:
            return None
        return rng.choice(self._inputs)

    def pick_by_energy(self, rng: random.Random) -> Optional[FuzzInput]:
        if not self._inputs:
            return None
        total_energy = sum(i.energy for i in self._inputs)
        if total_energy <= 0:
            return rng.choice(self._inputs)
        r = rng.uniform(0, total_energy)
        cumulative = 0.0
        for inp in self._inputs:
            cumulative += inp.energy
            if cumulative >= r:
                return inp
        return self._inputs[-1]

    @property
    def size(self) -> int:
        return len(self._inputs)

    @property
    def inputs(self) -> List[FuzzInput]:
        return list(self._inputs)


# ---------------------------------------------------------------------------
# Fuzz campaign
# ---------------------------------------------------------------------------

@dataclass
class FuzzCampaign:
    """A complete fuzzing campaign with results."""
    config: FuzzConfig
    divergences: List[FuzzResult] = field(default_factory=list)
    crashes: List[FuzzResult] = field(default_factory=list)
    stats: CampaignStats = field(default_factory=CampaignStats)
    corpus: Corpus = field(default_factory=Corpus)

    def summary(self) -> str:
        lines = [
            f"Fuzz Campaign Results:",
            f"  Total inputs:    {self.stats.total_inputs}",
            f"  Time:            {self.stats.total_time_ms / 1000:.1f}s",
            f"  Divergences:     {self.stats.divergences_found} ({self.stats.unique_divergences} unique)",
            f"  Crashes:         {self.stats.crashes_found} ({self.stats.unique_crashes} unique)",
            f"  Corpus size:     {self.stats.corpus_size}",
            f"  Coverage edges:  {self.stats.coverage_edges}",
            f"  Exec/sec:        {self.stats.executions_per_second:.1f}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fuzz engine
# ---------------------------------------------------------------------------

class FuzzEngine:
    """
    Main fuzzing engine for differential testing.
    
    Manages fuzzing campaigns with configurable input generation,
    coverage tracking, and divergence detection.
    """

    def __init__(
        self,
        config: Optional[FuzzConfig] = None,
        c_executor: Optional[Callable[[FuzzInput], Any]] = None,
        rust_executor: Optional[Callable[[FuzzInput], Any]] = None,
        comparator: Optional[Callable[[Any, Any], bool]] = None,
    ):
        self.config = config or FuzzConfig()
        self.c_executor = c_executor
        self.rust_executor = rust_executor
        self.comparator = comparator or (lambda a, b: a == b)
        self.generator = InputGenerator(self.config)

        # Coverage
        self._coverage_bitmap: bytearray = bytearray(65536)
        self._coverage_count = 0

    def run_campaign(
        self,
        param_types: Optional[List[Tuple[str, IRType]]] = None,
    ) -> FuzzCampaign:
        """Run a full fuzzing campaign."""
        campaign = FuzzCampaign(config=self.config)
        start_time = time.time()

        # Initialize corpus with seeds
        for seed in self.config.seed_inputs:
            campaign.corpus.add(seed)

        # Generate initial inputs if no seeds
        if campaign.corpus.size == 0:
            for _ in range(min(100, self.config.max_iterations // 10)):
                if param_types:
                    inp = self.generator.generate_typed(param_types)
                else:
                    inp = self.generator.generate_random()
                campaign.corpus.add(inp)

        # Main fuzzing loop
        iteration = 0
        while iteration < self.config.max_iterations:
            elapsed = time.time() - start_time
            if elapsed > self.config.max_time_seconds:
                break

            # Pick input from corpus and mutate
            base = campaign.corpus.pick_by_energy(self.generator.rng)
            if base is None:
                if param_types:
                    inp = self.generator.generate_typed(param_types)
                else:
                    inp = self.generator.generate_random()
            else:
                # Apply multiple mutations
                inp = base
                depth = self.generator.rng.randint(1, self.config.mutation_depth)
                for _ in range(depth):
                    inp = self.generator.mutate(inp)

            # Execute
            result = self._execute_input(inp)
            campaign.stats.total_inputs += 1

            # Check for divergence
            if result.is_divergent:
                campaign.divergences.append(result)
                campaign.stats.divergences_found += 1
                fp = result.input.fingerprint()
                campaign.stats.unique_divergences = len(
                    {r.input.fingerprint() for r in campaign.divergences}
                )

            if result.is_crash:
                campaign.crashes.append(result)
                campaign.stats.crashes_found += 1
                campaign.stats.unique_crashes = len(
                    {r.input.fingerprint() for r in campaign.crashes}
                )

            # Update corpus if interesting
            if result.is_interesting:
                inp.energy = 2.0  # Boost energy
                campaign.corpus.add(inp)

            iteration += 1

        total_time = (time.time() - start_time) * 1000
        campaign.stats.total_time_ms = total_time
        campaign.stats.corpus_size = campaign.corpus.size
        campaign.stats.coverage_edges = self._coverage_count
        if total_time > 0:
            campaign.stats.executions_per_second = (
                campaign.stats.total_inputs / (total_time / 1000)
            )

        return campaign

    def _execute_input(self, inp: FuzzInput) -> FuzzResult:
        """Execute a single input against both C and Rust implementations."""
        result = FuzzResult(input=inp)
        start = time.time()

        # Execute C side
        c_output = None
        if self.c_executor is not None:
            try:
                c_output = self.c_executor(inp)
                result.c_output = c_output
            except Exception as e:
                result.c_crashed = True
                result.c_error = str(e)

        # Execute Rust side
        rust_output = None
        if self.rust_executor is not None:
            try:
                rust_output = self.rust_executor(inp)
                result.rust_output = rust_output
            except Exception as e:
                result.rust_crashed = True
                result.rust_error = str(e)

        result.execution_time_ms = (time.time() - start) * 1000

        # Check divergence
        if c_output is not None and rust_output is not None:
            if not self.comparator(c_output, rust_output):
                result.is_divergent = True
                result.divergence_description = (
                    f"C={c_output}, Rust={rust_output}"
                )

        # Check crash divergence
        if result.c_crashed != result.rust_crashed:
            result.is_divergent = True
            crashed_side = "C" if result.c_crashed else "Rust"
            result.divergence_description = f"{crashed_side} crashed, other did not"

        return result

    def fuzz_single(self, inp: FuzzInput) -> FuzzResult:
        """Execute a single input and return result."""
        return self._execute_input(inp)

    def fuzz_batch(self, inputs: List[FuzzInput]) -> List[FuzzResult]:
        """Execute a batch of inputs."""
        return [self._execute_input(inp) for inp in inputs]

    def set_executors(
        self,
        c_executor: Callable[[FuzzInput], Any],
        rust_executor: Callable[[FuzzInput], Any],
    ) -> None:
        """Set the C and Rust executors."""
        self.c_executor = c_executor
        self.rust_executor = rust_executor

    def add_seed(self, inp: FuzzInput) -> None:
        """Add a seed input."""
        self.config.seed_inputs.append(inp)

    def add_dictionary_entry(self, data: bytes) -> None:
        """Add a dictionary entry for mutation."""
        self.config.dictionary.append(data)
