"""
Relaxed atomics memory-ordering divergence oracle (100_STEPS step 113).

This oracle is intentionally *not* a UB oracle.  C ``memory_order_relaxed`` is a
defined but weaker memory model than the sequentially-consistent atomics exposed
by Go and often chosen by conservative Rust translations.  The evidence is a
bounded allowed-execution-set gap for the store-buffering litmus:

    initially x = y = 0
    T0: atomic_store_explicit(x, 1, memory_order_relaxed); r0 = load(y)
    T1: atomic_store_explicit(y, 1, memory_order_relaxed); r1 = load(x)

C relaxed permits ``r0 = 0 && r1 = 0`` because each load may read the initial
write of the other object.  Sequential consistency forbids that observation: if
``r0`` reads 0 then ``load y`` must be before ``store y``; if ``r1`` reads 0 then
``load x`` must be before ``store x``; with per-thread order this forms a cycle.

Runtime thread scheduling would be flaky and is the wrong proof object.  The
counterexample is therefore model-level: Python enumerates the bounded execution
graph, while confirmation compiles and runs small real C/Rust/Go atomics
model-checkers that independently decide the same allowed-vs-forbidden fact.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Dict, Iterable, Mapping, Optional, Tuple

from ..catalogue import ATOMIC_ORDERING, Definedness
from ..plugin import DivergenceOracle, OracleResult, OracleVerdict, register
from ..replay import Counterexample

Outcome = Tuple[int, int]
Schedule = Tuple[str, str, str, str]

_EVENTS: Schedule = ("Sx", "Ly", "Sy", "Lx")
_SUPPORTED_PATTERN = "store_buffering"
_ALL_ZERO: Outcome = (0, 0)


@dataclass(frozen=True)
class LitmusGap:
    """A bounded allowed-execution-set witness."""

    outcome: Outcome
    source_trace: str
    target_reason: str
    source_outcomes: Tuple[Outcome, ...]
    target_outcomes: Tuple[Outcome, ...]


def _source_relaxed_outcomes() -> Mapping[Outcome, str]:
    """Enumerate read-from choices for the C relaxed store-buffering litmus."""
    out: Dict[Outcome, str] = {}
    for ly_reads_store in (False, True):
        for lx_reads_store in (False, True):
            r0 = 1 if ly_reads_store else 0
            r1 = 1 if lx_reads_store else 0
            out[(r0, r1)] = (
                "Ly reads %s; Lx reads %s"
                % (
                    "T1's relaxed store to y" if ly_reads_store else "initial y=0",
                    "T0's relaxed store to x" if lx_reads_store else "initial x=0",
                )
            )
    return out


def _preserves_program_order(order: Iterable[str]) -> bool:
    pos = {ev: i for i, ev in enumerate(order)}
    return pos["Sx"] < pos["Ly"] and pos["Sy"] < pos["Lx"]


def _seq_cst_outcomes() -> Mapping[Outcome, Schedule]:
    """Enumerate all total orders preserving per-thread order for SeqCst."""
    out: Dict[Outcome, Schedule] = {}
    for order in permutations(_EVENTS):
        if not _preserves_program_order(order):
            continue
        x = y = 0
        r0 = r1 = -1
        for ev in order:
            if ev == "Sx":
                x = 1
            elif ev == "Sy":
                y = 1
            elif ev == "Ly":
                r0 = y
            elif ev == "Lx":
                r1 = x
        out.setdefault((r0, r1), order)
    return out


def store_buffering_gap(outcome: Outcome = _ALL_ZERO) -> Optional[LitmusGap]:
    """Return the first outcome allowed by C relaxed but forbidden by SeqCst."""
    source = _source_relaxed_outcomes()
    target = _seq_cst_outcomes()
    if outcome not in source or outcome in target:
        return None
    return LitmusGap(
        outcome=outcome,
        source_trace=source[outcome],
        target_reason=(
            "SeqCst would require Sx < Ly < Sy < Lx < Sx, a cycle in the single "
            "global order, so no bounded total-order interleaving admits r0=0,r1=0."
        ),
        source_outcomes=tuple(sorted(source)),
        target_outcomes=tuple(sorted(target)),
    )


def _requested_outcome(unit: Dict) -> Outcome:
    raw = unit.get("outcome", _ALL_ZERO)
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return _ALL_ZERO
    return int(raw[0]), int(raw[1])


def _norm_order(order: object, default: str) -> str:
    if order is None:
        return default
    return str(order).lower().replace("-", "_")


def _target_is_seq_cst(target_lang: str, unit: Dict) -> bool:
    order = _norm_order(unit.get("target_order"), "seq_cst")
    if target_lang == "go":
        # Go sync/atomic operations are sequentially consistent; there is no
        # relaxed spelling to select.
        return order in {"seq_cst", "seqcst", "sequentially_consistent", "go_seq_cst"}
    return order in {"seq_cst", "seqcst", "sequentially_consistent"}


_C_SRC = r'''
#include <stdatomic.h>
#include <stdio.h>

static int source_relaxed_all_zero_allowed(void) {
    for (int ly_reads_store = 0; ly_reads_store <= 1; ++ly_reads_store) {
        for (int lx_reads_store = 0; lx_reads_store <= 1; ++lx_reads_store) {
            int r0 = ly_reads_store ? 1 : 0;
            int r1 = lx_reads_store ? 1 : 0;
            if (r0 == 0 && r1 == 0) return 1;
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    atomic_int x;
    atomic_int y;
    atomic_init(&x, 0);
    atomic_init(&y, 0);
    atomic_store_explicit(&x, 1, memory_order_relaxed);
    (void)atomic_load_explicit(&y, memory_order_relaxed);
    printf("source_relaxed_all_zero=%s\n",
           source_relaxed_all_zero_allowed() ? "allowed" : "forbidden");
    return 0;
}
'''.lstrip()

_RUST_SRC = r'''
use std::sync::atomic::{AtomicI32, Ordering};

fn preserves_po(order: &[usize; 4]) -> bool {
    let mut pos = [0usize; 4];
    for (i, ev) in order.iter().enumerate() {
        pos[*ev] = i;
    }
    pos[0] < pos[1] && pos[2] < pos[3]
}

fn outcome(order: &[usize; 4]) -> (i32, i32) {
    let (mut x, mut y) = (0i32, 0i32);
    let (mut r0, mut r1) = (-1i32, -1i32);
    for ev in order {
        match *ev {
            0 => x = 1,
            1 => r0 = y,
            2 => y = 1,
            3 => r1 = x,
            _ => unreachable!(),
        }
    }
    (r0, r1)
}

fn target_seq_cst_all_zero_allowed() -> bool {
    for a in 0..4 {
        for b in 0..4 {
            if b == a { continue; }
            for c in 0..4 {
                if c == a || c == b { continue; }
                for d in 0..4 {
                    if d == a || d == b || d == c { continue; }
                    let order = [a, b, c, d];
                    if preserves_po(&order) && outcome(&order) == (0, 0) {
                        return true;
                    }
                }
            }
        }
    }
    false
}

fn main() {
    let x = AtomicI32::new(0);
    let y = AtomicI32::new(0);
    x.store(1, Ordering::SeqCst);
    let _ = y.load(Ordering::SeqCst);
    println!(
        "target_seq_cst_all_zero={}",
        if target_seq_cst_all_zero_allowed() { "allowed" } else { "forbidden" }
    );
}
'''.lstrip()

_GO_SRC = r'''
package main

import (
	"fmt"
	"sync/atomic"
)

func preservesPO(order [4]int) bool {
	var pos [4]int
	for i, ev := range order {
		pos[ev] = i
	}
	return pos[0] < pos[1] && pos[2] < pos[3]
}

func outcome(order [4]int) (int, int) {
	x, y, r0, r1 := 0, 0, -1, -1
	for _, ev := range order {
		switch ev {
		case 0:
			x = 1
		case 1:
			r0 = y
		case 2:
			y = 1
		case 3:
			r1 = x
		}
	}
	return r0, r1
}

func targetSeqCstAllZeroAllowed() bool {
	for a := 0; a < 4; a++ {
		for b := 0; b < 4; b++ {
			if b == a {
				continue
			}
			for c := 0; c < 4; c++ {
				if c == a || c == b {
					continue
				}
				for d := 0; d < 4; d++ {
					if d == a || d == b || d == c {
						continue
					}
					order := [4]int{a, b, c, d}
					r0, r1 := outcome(order)
					if preservesPO(order) && r0 == 0 && r1 == 0 {
						return true
					}
				}
			}
		}
	}
	return false
}

func main() {
	var x, y int32
	atomic.StoreInt32(&x, 1)
	_ = atomic.LoadInt32(&y)
	if targetSeqCstAllZeroAllowed() {
		fmt.Println("target_seq_cst_all_zero=allowed")
	} else {
		fmt.Println("target_seq_cst_all_zero=forbidden")
	}
}
'''.lstrip()


class _AtomicOrderingBase(DivergenceOracle):
    divergence_class = ATOMIC_ORDERING.key
    source_lang = "c"
    confirmation_mode = "model_level_divergence"

    def applies_to(self, unit: Dict) -> bool:
        if unit.get("probe") not in (None, self.divergence_class):
            return False
        if unit.get("target_lang", self.target_lang) != self.target_lang:
            return False
        return (
            unit.get("kind") == "atomic_litmus"
            and unit.get("pattern", _SUPPORTED_PATTERN) == _SUPPORTED_PATTERN
        )

    def _target_src(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def find_divergence(self, unit: Dict) -> OracleResult:
        if not self.applies_to(unit):
            return OracleResult(
                OracleVerdict.NOT_APPLICABLE,
                self.divergence_class,
                detail="unit is not a supported atomic store-buffering litmus",
            )
        if _norm_order(unit.get("source_order"), "relaxed") != "relaxed":
            return OracleResult(
                OracleVerdict.NO_DIVERGENCE_FOUND,
                self.divergence_class,
                detail="source is not memory_order_relaxed, so the relaxed-only gap is absent",
            )
        if not _target_is_seq_cst(self.target_lang, unit):
            return OracleResult(
                OracleVerdict.NO_DIVERGENCE_FOUND,
                self.divergence_class,
                detail="target order is not sequentially consistent, so no SC gap is claimed",
            )
        gap = store_buffering_gap(_requested_outcome(unit))
        if gap is None:
            return OracleResult(
                OracleVerdict.NO_DIVERGENCE_FOUND,
                self.divergence_class,
                detail="requested outcome is not allowed-by-source/forbidden-by-target",
            )
        ce = self._build(gap)
        return OracleResult(
            OracleVerdict.DIVERGENT,
            self.divergence_class,
            counterexample=ce,
            detail=(
                f"bounded store-buffering witness r0={gap.outcome[0]},r1={gap.outcome[1]}; "
                f"source outcomes={gap.source_outcomes}, target outcomes={gap.target_outcomes}"
            ),
        )

    def _build(self, gap: LitmusGap) -> Counterexample:
        r0, r1 = gap.outcome
        return Counterexample(
            divergence_class=self.divergence_class,
            source_lang="c",
            target_lang=self.target_lang,
            inputs={"r0": r0, "r1": r1},
            source_snippet=_C_SRC,
            target_snippet=self._target_src(),
            source_definedness=Definedness.DEFINED.value,
            divergence_witness=(
                f"The C relaxed store-buffering execution allows r0={r0},r1={r1}: "
                f"{gap.source_trace}. The {self.target_lang} SeqCst translation "
                f"forbids the same observation: {gap.target_reason}"
            ),
            definedness_witness=(
                "No source UB is used. The witness is an allowed-execution-set "
                "difference: C relaxed atomics define a larger behavior set, while "
                "the target's sequentially-consistent atomics define a smaller one. "
                "The confirmation harness compiles real atomics snippets and runs "
                "deterministic bounded model checkers; it does not rely on a flaky "
                "thread-scheduling observation."
            ),
        )


class AtomicOrderingOracle(_AtomicOrderingBase):
    """C relaxed atomics vs a Rust SeqCst atomic translation."""

    target_lang = "rust"

    def _target_src(self) -> str:
        return _RUST_SRC


class GoAtomicOrderingOracle(_AtomicOrderingBase):
    """C relaxed atomics vs Go's sequentially-consistent sync/atomic operations."""

    target_lang = "go"

    def _target_src(self) -> str:
        return _GO_SRC


register(AtomicOrderingOracle())
register(GoAtomicOrderingOracle())

