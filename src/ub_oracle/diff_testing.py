"""
Differential-testing baseline + the "fuzzing gap" measurement (100_STEPS 22/24).

The central empirical claim is that random/differential testing *misses* the
UB-rooted divergences this oracle finds, because the triggering inputs are a
vanishing fraction of the input space.  This module makes that claim measurable
and honest rather than rhetorical.

For a signed-overflow unit ``f(x) = x <op> c`` at a given bit width, the set of
overflowing (UB-triggering) inputs has a known size, so we can compute exactly
the probability a uniform random sample hits one, and also *empirically* run a
seeded random fuzzer to show how rarely it does.  The oracle, by contrast, finds
the witness deterministically via Z3 (probability 1).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Optional


def _signed_bounds(width: int):
    return -(1 << (width - 1)), (1 << (width - 1)) - 1


def overflow_input_count(op: str, c: int, width: int) -> int:
    """Exact number of signed inputs x for which ``x <op> c`` overflows."""
    lo, hi = _signed_bounds(width)
    count = 0
    # Closed-form ranges (kept simple & obviously-correct rather than clever):
    if op == "add":
        # overflow iff x + c > hi (for c > 0) or x + c < lo (for c < 0)
        if c > 0:
            count = max(0, hi - (hi - c))  # x in (hi-c, hi]  -> c values (clamped)
            count = min(c, hi - lo + 1)
        elif c < 0:
            count = min(-c, hi - lo + 1)
        else:
            count = 0
    elif op == "sub":
        # x - c overflows: x - c < lo (c>0) or x - c > hi (c<0)
        if c > 0:
            count = min(c, hi - lo + 1)
        elif c < 0:
            count = min(-c, hi - lo + 1)
        else:
            count = 0
    return count


def _overflows(x: int, op: str, c: int, width: int) -> bool:
    lo, hi = _signed_bounds(width)
    real = x + c if op == "add" else x - c
    return real < lo or real > hi


@dataclass
class FuzzingGap:
    op: str
    const: int
    width: int
    total_inputs: int
    overflow_inputs: int
    hit_probability: float
    trials: int
    empirical_hits: int
    seed: int

    @property
    def oracle_hit_probability(self) -> float:
        # The Z3 oracle finds the witness deterministically.
        return 1.0 if self.overflow_inputs > 0 else 0.0

    def summary(self) -> str:
        return (
            f"x {self.op} {self.const} @w{self.width}: "
            f"{self.overflow_inputs}/{self.total_inputs} UB inputs "
            f"(p={self.hit_probability:.3e}); "
            f"fuzzer hit {self.empirical_hits}/{self.trials}; "
            f"oracle p={self.oracle_hit_probability:.0f}"
        )


def measure_fuzzing_gap(op: str, c: int, width: int,
                        trials: int = 100_000, seed: int = 0) -> FuzzingGap:
    """Run a seeded uniform random fuzzer and compare to the analytic probability."""
    lo, hi = _signed_bounds(width)
    total = hi - lo + 1
    n_of = overflow_input_count(op, c, width)
    rng = random.Random(seed)
    hits = 0
    for _ in range(trials):
        x = rng.randint(lo, hi)
        if _overflows(x, op, c, width):
            hits += 1
    return FuzzingGap(
        op=op, const=c, width=width,
        total_inputs=total, overflow_inputs=n_of,
        hit_probability=n_of / total if total else 0.0,
        trials=trials, empirical_hits=hits, seed=seed,
    )
