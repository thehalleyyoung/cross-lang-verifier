"""
Data flow analysis framework for the Cross-Language Equivalence Verifier.

Provides a generic worklist-based dataflow framework (forward/backward,
may/must) along with instantiations for: reaching definitions, live
variables, available expressions, very busy expressions, constant
propagation (sparse conditional), def-use chains, and use-def chains.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
)

from ..ir.basic_block import BasicBlock
from ..ir.function import Function
from ..ir.instructions import (
    Instruction,
    BinaryOp,
    UnaryOp,
    CompareOp,
    CastInst,
    LoadInst,
    StoreInst,
    AllocaInst,
    CallInst,
    ReturnInst,
    PhiInst,
    SelectInst,
    Value,
    Constant,
    Argument,
    BinOpKind,
)


# ── Direction enum ───────────────────────────────────────────────────────

class DataflowDirection(Enum):
    FORWARD = auto()
    BACKWARD = auto()


class MeetKind(Enum):
    MAY = auto()   # Union (any path)
    MUST = auto()  # Intersection (all paths)


# ── Generic dataflow result ─────────────────────────────────────────────

T = TypeVar("T")


@dataclass
class DataflowResult(Generic[T]):
    """Result of a dataflow analysis, mapping blocks to IN/OUT sets."""
    block_in: dict[int, T] = field(default_factory=dict)
    block_out: dict[int, T] = field(default_factory=dict)
    iterations: int = 0

    def get_in(self, block: BasicBlock) -> T:
        return self.block_in[block.id]

    def get_out(self, block: BasicBlock) -> T:
        return self.block_out[block.id]

    def summary(self) -> str:
        lines = [f"Dataflow result ({self.iterations} iterations):"]
        for bid in sorted(self.block_in.keys()):
            lines.append(f"  block_{bid}: IN={self.block_in[bid]}")
            lines.append(f"           OUT={self.block_out[bid]}")
        return "\n".join(lines)


# ── Generic dataflow analysis ────────────────────────────────────────────

class DataflowAnalysis(ABC, Generic[T]):
    """Abstract base for a dataflow analysis.

    Subclasses define the lattice domain, transfer function, meet operator,
    and initial/boundary values.  The framework runs the iterative worklist
    algorithm to convergence.
    """

    @abstractmethod
    def direction(self) -> DataflowDirection:
        ...

    @abstractmethod
    def initial_value(self) -> T:
        """Value for non-entry (or non-exit) blocks at start of iteration."""
        ...

    @abstractmethod
    def boundary_value(self) -> T:
        """Value for the entry (forward) or exit (backward) block."""
        ...

    @abstractmethod
    def meet(self, values: Sequence[T]) -> T:
        """Combine values from multiple predecessors/successors."""
        ...

    @abstractmethod
    def transfer(self, block: BasicBlock, in_value: T) -> T:
        """Transfer function: compute OUT from IN (forward) or IN from OUT (backward)."""
        ...

    def analyze(self, function: Function) -> DataflowResult[T]:
        """Run the analysis to a fixed point."""
        blocks = function.iter_blocks_rpo()
        if not blocks:
            return DataflowResult()

        result = DataflowResult[T]()
        is_forward = self.direction() is DataflowDirection.FORWARD

        # Initialize
        entry = blocks[0]
        for block in blocks:
            result.block_in[block.id] = self.initial_value()
            result.block_out[block.id] = self.initial_value()

        if is_forward:
            result.block_in[entry.id] = self.boundary_value()
            result.block_out[entry.id] = self.transfer(entry, result.block_in[entry.id])
        else:
            # For backward analysis, initialize exit blocks
            for block in blocks:
                if block.is_exit:
                    result.block_out[block.id] = self.boundary_value()
                    result.block_in[block.id] = self.transfer(block, result.block_out[block.id])

        # Worklist iteration
        worklist: deque[BasicBlock] = deque()
        in_worklist: set[int] = set()

        if is_forward:
            order = blocks[1:]  # Skip entry
        else:
            order = list(reversed(blocks))
            # Skip exit blocks for initial worklist
            order = [b for b in order if not b.is_exit]

        for block in order:
            worklist.append(block)
            in_worklist.add(block.id)

        iterations = 0
        max_iterations = len(blocks) * 100  # Safety limit

        while worklist and iterations < max_iterations:
            block = worklist.popleft()
            in_worklist.discard(block.id)
            iterations += 1

            if is_forward:
                # IN = meet of predecessor OUTs
                preds = block.predecessors
                if preds:
                    pred_outs = [result.block_out[p.id] for p in preds]
                    new_in = self.meet(pred_outs)
                else:
                    new_in = self.boundary_value()

                result.block_in[block.id] = new_in
                new_out = self.transfer(block, new_in)

                if new_out != result.block_out[block.id]:
                    result.block_out[block.id] = new_out
                    for succ in block.successors:
                        if succ.id not in in_worklist:
                            worklist.append(succ)
                            in_worklist.add(succ.id)
            else:
                # OUT = meet of successor INs
                succs = block.successors
                if succs:
                    succ_ins = [result.block_in[s.id] for s in succs]
                    new_out = self.meet(succ_ins)
                else:
                    new_out = self.boundary_value()

                result.block_out[block.id] = new_out
                new_in = self.transfer(block, new_out)

                if new_in != result.block_in[block.id]:
                    result.block_in[block.id] = new_in
                    for pred in block.predecessors:
                        if pred.id not in in_worklist:
                            worklist.append(pred)
                            in_worklist.add(pred.id)

        result.iterations = iterations
        return result


# ── Reaching Definitions ────────────────────────────────────────────────

class ReachingDefinitions(DataflowAnalysis[frozenset[int]]):
    """Forward may-analysis: which definitions reach each program point.

    Each element of the set is the Value.id of an instruction that defines
    a value.
    """

    def __init__(self, function: Function) -> None:
        self.function = function
        # Pre-compute gen/kill sets
        self._gen: dict[int, frozenset[int]] = {}
        self._kill: dict[int, frozenset[int]] = {}
        self._all_defs: dict[str, set[int]] = {}  # name → set of defining inst ids
        self._precompute(function)

    def _precompute(self, function: Function) -> None:
        # Collect all definitions by name
        for block in function.blocks:
            for inst in block:
                if inst.name:
                    self._all_defs.setdefault(inst.name, set()).add(inst.id)

        for block in function.blocks:
            gen: set[int] = set()
            kill: set[int] = set()
            for inst in block:
                if inst.name:
                    # Kill all other defs of the same name
                    for other_id in self._all_defs.get(inst.name, set()):
                        if other_id != inst.id:
                            kill.add(other_id)
                    # Gen this def
                    gen.add(inst.id)
            self._gen[block.id] = frozenset(gen)
            self._kill[block.id] = frozenset(kill)

    def direction(self) -> DataflowDirection:
        return DataflowDirection.FORWARD

    def initial_value(self) -> frozenset[int]:
        return frozenset()

    def boundary_value(self) -> frozenset[int]:
        return frozenset()

    def meet(self, values: Sequence[frozenset[int]]) -> frozenset[int]:
        # Union (may analysis)
        result: set[int] = set()
        for v in values:
            result |= v
        return frozenset(result)

    def transfer(self, block: BasicBlock, in_value: frozenset[int]) -> frozenset[int]:
        # OUT = GEN ∪ (IN - KILL)
        gen = self._gen.get(block.id, frozenset())
        kill = self._kill.get(block.id, frozenset())
        return gen | (in_value - kill)

    def run(self) -> DataflowResult[frozenset[int]]:
        return self.analyze(self.function)


# ── Live Variables ───────────────────────────────────────────────────────

class LiveVariables(DataflowAnalysis[frozenset[int]]):
    """Backward may-analysis: which variables are live at each point.

    A variable is live at a point if it may be used before being redefined.
    """

    def __init__(self, function: Function) -> None:
        self.function = function
        self._use: dict[int, frozenset[int]] = {}
        self._def: dict[int, frozenset[int]] = {}
        self._precompute(function)

    def _precompute(self, function: Function) -> None:
        for block in function.blocks:
            uses: set[int] = set()
            defs: set[int] = set()
            # Process instructions in order; a use before def makes it "upward exposed"
            for inst in block:
                for op in inst.operands:
                    if not isinstance(op, Constant) and op.id not in defs:
                        uses.add(op.id)
                # The instruction itself is a definition
                from ..ir.types import VoidType
                if not isinstance(inst.type, VoidType):
                    defs.add(inst.id)
            self._use[block.id] = frozenset(uses)
            self._def[block.id] = frozenset(defs)

    def direction(self) -> DataflowDirection:
        return DataflowDirection.BACKWARD

    def initial_value(self) -> frozenset[int]:
        return frozenset()

    def boundary_value(self) -> frozenset[int]:
        return frozenset()

    def meet(self, values: Sequence[frozenset[int]]) -> frozenset[int]:
        result: set[int] = set()
        for v in values:
            result |= v
        return frozenset(result)

    def transfer(self, block: BasicBlock, out_value: frozenset[int]) -> frozenset[int]:
        # IN = USE ∪ (OUT - DEF)
        use = self._use.get(block.id, frozenset())
        deff = self._def.get(block.id, frozenset())
        return use | (out_value - deff)

    def run(self) -> DataflowResult[frozenset[int]]:
        return self.analyze(self.function)


# ── Available Expressions ────────────────────────────────────────────────

@dataclass(frozen=True)
class Expression:
    """An expression represented as (opcode, operand_id_1, operand_id_2)."""
    opcode: str
    operand_ids: tuple[int, ...]

    def __str__(self) -> str:
        ops = ", ".join(str(oid) for oid in self.operand_ids)
        return f"{self.opcode}({ops})"


class AvailableExpressions(DataflowAnalysis[frozenset[Expression]]):
    """Forward must-analysis: which expressions are available (already computed).

    An expression is available at a point if it has been computed on every
    path from the entry to that point, and none of its operands have been
    redefined since.
    """

    def __init__(self, function: Function) -> None:
        self.function = function
        self._gen: dict[int, frozenset[Expression]] = {}
        self._kill: dict[int, frozenset[Expression]] = {}
        self._all_exprs: frozenset[Expression] = frozenset()
        self._precompute(function)

    def _precompute(self, function: Function) -> None:
        all_exprs: set[Expression] = set()

        # Collect all expressions
        for block in function.blocks:
            for inst in block:
                expr = self._to_expression(inst)
                if expr is not None:
                    all_exprs.add(expr)
        self._all_exprs = frozenset(all_exprs)

        # Compute gen/kill per block
        for block in function.blocks:
            gen: set[Expression] = set()
            kill: set[Expression] = set()
            defined_in_block: set[int] = set()

            for inst in block:
                expr = self._to_expression(inst)
                # Kill expressions that use a value defined by this instruction
                for e in all_exprs:
                    if inst.id in e.operand_ids:
                        kill.add(e)
                # Gen this expression (if not killed)
                if expr is not None and expr not in kill:
                    gen.add(expr)
                defined_in_block.add(inst.id)

            self._gen[block.id] = frozenset(gen)
            self._kill[block.id] = frozenset(kill)

    def _to_expression(self, inst: Instruction) -> Expression | None:
        if isinstance(inst, BinaryOp):
            return Expression(inst.op.value, (inst.lhs.id, inst.rhs.id))
        if isinstance(inst, UnaryOp):
            return Expression(inst.op.value, (inst.operands[0].id,))
        if isinstance(inst, CompareOp):
            return Expression(
                f"cmp.{inst.predicate.value}",
                (inst.lhs.id, inst.rhs.id),
            )
        return None

    def direction(self) -> DataflowDirection:
        return DataflowDirection.FORWARD

    def initial_value(self) -> frozenset[Expression]:
        return self._all_exprs  # Must: start with all (intersection identity)

    def boundary_value(self) -> frozenset[Expression]:
        return frozenset()  # Nothing available at entry

    def meet(self, values: Sequence[frozenset[Expression]]) -> frozenset[Expression]:
        # Intersection (must analysis)
        if not values:
            return self._all_exprs
        result = set(values[0])
        for v in values[1:]:
            result &= v
        return frozenset(result)

    def transfer(
        self, block: BasicBlock, in_value: frozenset[Expression],
    ) -> frozenset[Expression]:
        gen = self._gen.get(block.id, frozenset())
        kill = self._kill.get(block.id, frozenset())
        return gen | (in_value - kill)

    def run(self) -> DataflowResult[frozenset[Expression]]:
        return self.analyze(self.function)


# ── Very Busy Expressions ───────────────────────────────────────────────

class VeryBusyExpressions(DataflowAnalysis[frozenset[Expression]]):
    """Backward must-analysis: expressions that will be used on every path
    from the current point before any operand is redefined.

    Useful for code hoisting optimizations.
    """

    def __init__(self, function: Function) -> None:
        self.function = function
        self._use: dict[int, frozenset[Expression]] = {}
        self._kill: dict[int, frozenset[Expression]] = {}
        self._all_exprs: frozenset[Expression] = frozenset()
        self._precompute(function)

    def _precompute(self, function: Function) -> None:
        all_exprs: set[Expression] = set()
        for block in function.blocks:
            for inst in block:
                expr = self._to_expression(inst)
                if expr is not None:
                    all_exprs.add(expr)
        self._all_exprs = frozenset(all_exprs)

        for block in function.blocks:
            use: set[Expression] = set()
            kill: set[Expression] = set()

            # Process in reverse order for backward analysis
            for inst in reversed(list(block)):
                # Kill expressions whose operands are defined here
                for e in all_exprs:
                    if inst.id in e.operand_ids:
                        kill.add(e)
                        use.discard(e)
                # Use this expression
                expr = self._to_expression(inst)
                if expr is not None:
                    use.add(expr)

            self._use[block.id] = frozenset(use)
            self._kill[block.id] = frozenset(kill)

    def _to_expression(self, inst: Instruction) -> Expression | None:
        if isinstance(inst, BinaryOp):
            return Expression(inst.op.value, (inst.lhs.id, inst.rhs.id))
        if isinstance(inst, UnaryOp):
            return Expression(inst.op.value, (inst.operands[0].id,))
        return None

    def direction(self) -> DataflowDirection:
        return DataflowDirection.BACKWARD

    def initial_value(self) -> frozenset[Expression]:
        return self._all_exprs

    def boundary_value(self) -> frozenset[Expression]:
        return frozenset()

    def meet(self, values: Sequence[frozenset[Expression]]) -> frozenset[Expression]:
        if not values:
            return self._all_exprs
        result = set(values[0])
        for v in values[1:]:
            result &= v
        return frozenset(result)

    def transfer(
        self, block: BasicBlock, out_value: frozenset[Expression],
    ) -> frozenset[Expression]:
        use = self._use.get(block.id, frozenset())
        kill = self._kill.get(block.id, frozenset())
        return use | (out_value - kill)

    def run(self) -> DataflowResult[frozenset[Expression]]:
        return self.analyze(self.function)


# ── Constant Propagation (Sparse Conditional) ───────────────────────────

class ConstantLattice(Enum):
    """Lattice for constant propagation."""
    TOP = auto()      # Unknown / not yet determined
    BOTTOM = auto()   # Overdefined (known to be non-constant)
    # Actual constant values are represented as (CONSTANT, value) tuples


@dataclass
class ConstantValue:
    """A value in the constant propagation lattice."""
    kind: ConstantLattice | None = None
    value: Any = None

    @staticmethod
    def top() -> "ConstantValue":
        return ConstantValue(kind=ConstantLattice.TOP)

    @staticmethod
    def bottom() -> "ConstantValue":
        return ConstantValue(kind=ConstantLattice.BOTTOM)

    @staticmethod
    def constant(v: Any) -> "ConstantValue":
        return ConstantValue(kind=None, value=v)

    @property
    def is_top(self) -> bool:
        return self.kind is ConstantLattice.TOP

    @property
    def is_bottom(self) -> bool:
        return self.kind is ConstantLattice.BOTTOM

    @property
    def is_constant(self) -> bool:
        return self.kind is None and self.value is not None

    def meet(self, other: "ConstantValue") -> "ConstantValue":
        if self.is_top:
            return other
        if other.is_top:
            return self
        if self.is_bottom or other.is_bottom:
            return ConstantValue.bottom()
        if self.value == other.value:
            return self
        return ConstantValue.bottom()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ConstantValue):
            return NotImplemented
        return self.kind == other.kind and self.value == other.value

    def __hash__(self) -> int:
        return hash((self.kind, self.value))

    def __repr__(self) -> str:
        if self.is_top:
            return "⊤"
        if self.is_bottom:
            return "⊥"
        return f"const({self.value})"


class ConstantPropagation:
    """Sparse conditional constant propagation (SCCP).

    Uses a worklist over SSA edges to propagate constants efficiently.
    """

    def __init__(self, function: Function) -> None:
        self.function = function
        self._lattice: dict[int, ConstantValue] = {}
        self._executable_edges: set[Tuple[int, int]] = set()
        self._executable_blocks: set[int] = set()
        self._ssa_worklist: deque[int] = deque()  # Value ids to process
        self._cfg_worklist: deque[BasicBlock] = deque()

    def run(self) -> dict[int, ConstantValue]:
        """Run SCCP and return the mapping from Value.id → ConstantValue."""
        # Initialize all values to TOP
        for block in self.function.blocks:
            for inst in block:
                self._lattice[inst.id] = ConstantValue.top()
        for arg in self.function.arguments:
            self._lattice[arg.id] = ConstantValue.bottom()  # Arguments are unknown

        # Seed: entry block is executable
        entry = self.function.entry_block
        if entry is None:
            return self._lattice

        self._cfg_worklist.append(entry)
        self._executable_blocks.add(entry.id)

        # Main loop
        while self._cfg_worklist or self._ssa_worklist:
            while self._cfg_worklist:
                block = self._cfg_worklist.popleft()
                self._visit_block(block)

            while self._ssa_worklist:
                val_id = self._ssa_worklist.popleft()
                self._visit_uses(val_id)

        return self._lattice

    def _visit_block(self, block: BasicBlock) -> None:
        """Process all instructions in a newly-executable block."""
        for inst in block:
            self._evaluate_instruction(inst)

    def _visit_uses(self, val_id: int) -> None:
        """Re-evaluate all users of a value whose lattice changed."""
        # Find the instruction/value
        for block in self.function.blocks:
            for inst in block:
                if inst.id == val_id:
                    for user in inst.users:
                        if isinstance(user, Instruction):
                            if user.parent and user.parent.id in self._executable_blocks:
                                self._evaluate_instruction(user)
                    return

    def _evaluate_instruction(self, inst: Instruction) -> None:
        """Evaluate an instruction under the current lattice."""
        old = self._lattice.get(inst.id, ConstantValue.top())

        if isinstance(inst, PhiInst):
            new = self._eval_phi(inst)
        elif isinstance(inst, BinaryOp):
            new = self._eval_binop(inst)
        elif isinstance(inst, UnaryOp):
            new = self._eval_unop(inst)
        elif isinstance(inst, CastInst):
            new = self._eval_cast(inst)
        elif isinstance(inst, CompareOp):
            new = self._eval_compare(inst)
        elif isinstance(inst, SelectInst):
            new = self._eval_select(inst)
        elif isinstance(inst, (LoadInst, StoreInst, CallInst)):
            new = ConstantValue.bottom()  # Conservative
        else:
            new = ConstantValue.bottom()

        merged = old.meet(new)
        if merged != old:
            self._lattice[inst.id] = merged
            self._ssa_worklist.append(inst.id)

        # Handle control flow: add executable successors
        if inst.parent and inst == inst.parent.terminator:
            self._propagate_cfg(inst)

    def _get_lattice(self, val: Value) -> ConstantValue:
        if isinstance(val, Constant):
            return ConstantValue.constant(val.value)
        return self._lattice.get(val.id, ConstantValue.top())

    def _eval_phi(self, inst: PhiInst) -> ConstantValue:
        result = ConstantValue.top()
        for val, block in inst.incoming:
            edge_key = (block.id, inst.parent.id) if inst.parent else None
            if edge_key and edge_key not in self._executable_edges:
                if block.id not in self._executable_blocks:
                    continue  # Skip non-executable predecessors
            lat = self._get_lattice(val)
            result = result.meet(lat)
        return result

    def _eval_binop(self, inst: BinaryOp) -> ConstantValue:
        lhs = self._get_lattice(inst.lhs)
        rhs = self._get_lattice(inst.rhs)

        if lhs.is_bottom or rhs.is_bottom:
            return ConstantValue.bottom()
        if lhs.is_top or rhs.is_top:
            return ConstantValue.top()

        lv, rv = lhs.value, rhs.value
        if not isinstance(lv, (int, float)) or not isinstance(rv, (int, float)):
            return ConstantValue.bottom()

        try:
            op = inst.op
            if op is BinOpKind.ADD:
                return ConstantValue.constant(lv + rv)
            if op is BinOpKind.SUB:
                return ConstantValue.constant(lv - rv)
            if op is BinOpKind.MUL:
                return ConstantValue.constant(lv * rv)
            if op in (BinOpKind.SDIV, BinOpKind.UDIV):
                if rv == 0:
                    return ConstantValue.bottom()
                return ConstantValue.constant(int(lv) // int(rv))
            if op in (BinOpKind.SREM, BinOpKind.UREM):
                if rv == 0:
                    return ConstantValue.bottom()
                return ConstantValue.constant(int(lv) % int(rv))
            if op is BinOpKind.AND:
                return ConstantValue.constant(int(lv) & int(rv))
            if op is BinOpKind.OR:
                return ConstantValue.constant(int(lv) | int(rv))
            if op is BinOpKind.XOR:
                return ConstantValue.constant(int(lv) ^ int(rv))
            if op is BinOpKind.SHL:
                return ConstantValue.constant(int(lv) << int(rv))
            if op is BinOpKind.LSHR:
                return ConstantValue.constant(int(lv) >> int(rv))
            if op is BinOpKind.ASHR:
                return ConstantValue.constant(int(lv) >> int(rv))
            if op is BinOpKind.FADD:
                return ConstantValue.constant(float(lv) + float(rv))
            if op is BinOpKind.FSUB:
                return ConstantValue.constant(float(lv) - float(rv))
            if op is BinOpKind.FMUL:
                return ConstantValue.constant(float(lv) * float(rv))
            if op is BinOpKind.FDIV:
                if float(rv) == 0.0:
                    return ConstantValue.bottom()
                return ConstantValue.constant(float(lv) / float(rv))
        except Exception:
            return ConstantValue.bottom()

        return ConstantValue.bottom()

    def _eval_unop(self, inst: UnaryOp) -> ConstantValue:
        operand = self._get_lattice(inst.operands[0])
        if operand.is_bottom:
            return ConstantValue.bottom()
        if operand.is_top:
            return ConstantValue.top()

        v = operand.value
        from ..ir.instructions import UnaryOpKind
        if inst.op is UnaryOpKind.NEG:
            return ConstantValue.constant(-v)
        if inst.op is UnaryOpKind.NOT:
            return ConstantValue.constant(int(not v))
        if inst.op is UnaryOpKind.BITWISE_NOT:
            return ConstantValue.constant(~int(v))
        if inst.op is UnaryOpKind.FNEG:
            return ConstantValue.constant(-float(v))
        return ConstantValue.bottom()

    def _eval_cast(self, inst: CastInst) -> ConstantValue:
        operand = self._get_lattice(inst.operand)
        if operand.is_bottom:
            return ConstantValue.bottom()
        if operand.is_top:
            return ConstantValue.top()
        # Conservative: just propagate the value
        return ConstantValue.constant(operand.value)

    def _eval_compare(self, inst: CompareOp) -> ConstantValue:
        lhs = self._get_lattice(inst.lhs)
        rhs = self._get_lattice(inst.rhs)
        if lhs.is_bottom or rhs.is_bottom:
            return ConstantValue.bottom()
        if lhs.is_top or rhs.is_top:
            return ConstantValue.top()

        from ..ir.instructions import CmpPredicate
        lv, rv = lhs.value, rhs.value
        pred = inst.predicate
        try:
            if pred is CmpPredicate.EQ:
                return ConstantValue.constant(int(lv == rv))
            if pred is CmpPredicate.NE:
                return ConstantValue.constant(int(lv != rv))
            if pred in (CmpPredicate.SLT, CmpPredicate.ULT, CmpPredicate.OLT):
                return ConstantValue.constant(int(lv < rv))
            if pred in (CmpPredicate.SLE, CmpPredicate.ULE, CmpPredicate.OLE):
                return ConstantValue.constant(int(lv <= rv))
            if pred in (CmpPredicate.SGT, CmpPredicate.UGT, CmpPredicate.OGT):
                return ConstantValue.constant(int(lv > rv))
            if pred in (CmpPredicate.SGE, CmpPredicate.UGE, CmpPredicate.OGE):
                return ConstantValue.constant(int(lv >= rv))
        except Exception:
            return ConstantValue.bottom()

        return ConstantValue.bottom()

    def _eval_select(self, inst: SelectInst) -> ConstantValue:
        cond = self._get_lattice(inst.condition)
        if cond.is_bottom:
            return ConstantValue.bottom()
        if cond.is_top:
            return ConstantValue.top()
        if cond.value:
            return self._get_lattice(inst.true_value)
        return self._get_lattice(inst.false_value)

    def _propagate_cfg(self, inst: Instruction) -> None:
        """Add successor blocks to the CFG worklist based on control flow."""
        block = inst.parent
        if block is None:
            return

        from ..ir.instructions import BranchInst, SwitchInst
        if isinstance(inst, BranchInst):
            if inst.condition is None:
                # Unconditional
                target = inst._true_target
                self._mark_executable(block, target)
            else:
                cond = self._get_lattice(inst.condition)
                if cond.is_constant:
                    # Only one branch is executable
                    if cond.value:
                        self._mark_executable(block, inst._true_target)
                    else:
                        if inst._false_target:
                            self._mark_executable(block, inst._false_target)
                else:
                    # Both branches executable
                    self._mark_executable(block, inst._true_target)
                    if inst._false_target:
                        self._mark_executable(block, inst._false_target)
        elif isinstance(inst, SwitchInst):
            # Conservative: all targets executable
            self._mark_executable(block, inst._default_target)
            for _, target in inst._cases:
                self._mark_executable(block, target)
        elif isinstance(inst, ReturnInst):
            pass  # No successors

    def _mark_executable(self, src: BasicBlock, dst: BasicBlock) -> None:
        edge_key = (src.id, dst.id)
        if edge_key not in self._executable_edges:
            self._executable_edges.add(edge_key)
            if dst.id not in self._executable_blocks:
                self._executable_blocks.add(dst.id)
                self._cfg_worklist.append(dst)
            else:
                # Block already visited but new edge; re-evaluate phis
                for phi in dst.phi_nodes:
                    self._evaluate_instruction(phi)


# ── Def-Use Chains ───────────────────────────────────────────────────────

@dataclass
class DefUseChains:
    """Maps each definition (Value.id) to its set of uses (Instruction objects)."""
    _chains: dict[int, set[Instruction]] = field(default_factory=dict)

    def uses_of(self, value: Value) -> set[Instruction]:
        return self._chains.get(value.id, set())

    def num_uses(self, value: Value) -> int:
        return len(self._chains.get(value.id, set()))

    def is_dead(self, value: Value) -> bool:
        return self.num_uses(value) == 0

    @classmethod
    def build(cls, function: Function) -> "DefUseChains":
        chains = cls()
        for block in function.blocks:
            for inst in block:
                for op in inst.operands:
                    chains._chains.setdefault(op.id, set()).add(inst)
        return chains

    def summary(self) -> str:
        lines = [f"Def-Use chains ({len(self._chains)} definitions):"]
        for vid, uses in sorted(self._chains.items()):
            use_strs = [u.opcode_name() for u in uses]
            lines.append(f"  %{vid} → [{', '.join(use_strs)}] ({len(uses)} uses)")
        return "\n".join(lines)


# ── Use-Def Chains ───────────────────────────────────────────────────────

@dataclass
class UseDefChains:
    """Maps each use (instruction operand position) to its defining instruction."""
    _chains: dict[Tuple[int, int], Value] = field(default_factory=dict)
    _inst_defs: dict[int, list[Value]] = field(default_factory=dict)

    def definition_of(self, inst: Instruction, operand_idx: int) -> Value | None:
        """Return the Value that defines operand operand_idx of inst."""
        return self._chains.get((inst.id, operand_idx))

    def all_definitions_for(self, inst: Instruction) -> list[Value]:
        return self._inst_defs.get(inst.id, [])

    @classmethod
    def build(cls, function: Function) -> "UseDefChains":
        chains = cls()
        for block in function.blocks:
            for inst in block:
                defs: list[Value] = []
                for i, op in enumerate(inst.operands):
                    chains._chains[(inst.id, i)] = op
                    defs.append(op)
                chains._inst_defs[inst.id] = defs
        return chains

    def summary(self) -> str:
        lines = [f"Use-Def chains ({len(self._chains)} uses):"]
        for (inst_id, op_idx), defn in sorted(self._chains.items()):
            lines.append(f"  inst %{inst_id} op[{op_idx}] ← {defn.display_name}")
        return "\n".join(lines)
