"""
Global Value Numbering for the Cross-Language Equivalence Verifier.

Implements hash-based value numbering with congruence detection
for redundant computation elimination. Includes load elimination
via memory dependence tracking.

Provides:
- GlobalValueNumbering: main GVN pass
- ValueTable: hash-consing table for value expressions
- CongruenceClass: groups of congruent values
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from ...ir.function import Function
from ...ir.basic_block import BasicBlock
from ...ir.instructions import (
    Instruction, Value, Constant, Argument,
    BinaryOp, BinOpKind, UnaryOp, UnaryOpKind,
    CompareOp, CmpPredicate,
    LoadInst, StoreInst, AllocaInst,
    GetElementPtrInst, CastInst, CastKind,
    CallInst, ReturnInst, BranchInst,
    PhiInst, SelectInst,
)
from ...ir.types import IRType, IntType, FloatType
from .pass_manager import FunctionPass, PassResult, AnalysisManager

logger = logging.getLogger(__name__)


# ─── Value Expression ──────────────────────────────────────────────────

class ExprKind(Enum):
    """Kind of value expression in the value table."""
    CONSTANT = auto()
    ARGUMENT = auto()
    BINARY = auto()
    UNARY = auto()
    COMPARE = auto()
    CAST = auto()
    SELECT = auto()
    PHI = auto()
    LOAD = auto()
    GEP = auto()
    CALL = auto()
    UNKNOWN = auto()


@dataclass(frozen=True)
class ValueExpression:
    """Hashable representation of a value computation.

    Two instructions that compute the same expression (same operation
    on same value numbers) will have the same ValueExpression.
    """
    kind: ExprKind
    opcode: Any = None
    operands: Tuple[int, ...] = ()
    type_hash: int = 0
    extra: Any = None

    def __hash__(self) -> int:
        return hash((self.kind, self.opcode, self.operands, self.type_hash, self.extra))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ValueExpression):
            return NotImplemented
        return (self.kind == other.kind and
                self.opcode == other.opcode and
                self.operands == other.operands and
                self.type_hash == other.type_hash and
                self.extra == other.extra)


# ─── Congruence Class ──────────────────────────────────────────────────

class CongruenceClass:
    """A set of values that are all congruent (compute the same value).

    The 'leader' is the representative value that other members
    should be replaced with.
    """

    def __init__(self, value_number: int, leader: Value) -> None:
        self._vn = value_number
        self._leader = leader
        self._members: Set[int] = {leader.id if hasattr(leader, 'id') else id(leader)}
        self._expression: Optional[ValueExpression] = None

    @property
    def value_number(self) -> int:
        return self._vn

    @property
    def leader(self) -> Value:
        return self._leader

    @leader.setter
    def leader(self, val: Value) -> None:
        self._leader = val

    @property
    def members(self) -> Set[int]:
        return self._members

    @property
    def expression(self) -> Optional[ValueExpression]:
        return self._expression

    @expression.setter
    def expression(self, expr: ValueExpression) -> None:
        self._expression = expr

    def add_member(self, val_id: int) -> None:
        self._members.add(val_id)

    def remove_member(self, val_id: int) -> None:
        self._members.discard(val_id)

    @property
    def size(self) -> int:
        return len(self._members)

    def __str__(self) -> str:
        return f"CC(vn={self._vn}, leader={self._leader}, members={len(self._members)})"


# ─── Value Table ───────────────────────────────────────────────────────

class ValueTable:
    """Hash-consing table mapping value expressions to value numbers.

    Maintains a mapping from ValueExpression → value_number and
    from value_id → value_number. Two values with the same
    expression receive the same value number.
    """

    def __init__(self) -> None:
        self._next_vn: int = 0
        self._expr_to_vn: Dict[ValueExpression, int] = {}
        self._value_to_vn: Dict[int, int] = {}
        self._vn_to_class: Dict[int, CongruenceClass] = {}
        self._value_to_value: Dict[int, Value] = {}

    def lookup_or_add(self, val: Value, expr: ValueExpression) -> int:
        """Look up or create a value number for an expression."""
        val_id = val.id if hasattr(val, 'id') else id(val)

        # Check if expression already has a value number
        if expr in self._expr_to_vn:
            vn = self._expr_to_vn[expr]
            self._value_to_vn[val_id] = vn
            self._value_to_value[val_id] = val
            cc = self._vn_to_class.get(vn)
            if cc is not None:
                cc.add_member(val_id)
            return vn

        # Create new value number
        vn = self._next_vn
        self._next_vn += 1
        self._expr_to_vn[expr] = vn
        self._value_to_vn[val_id] = vn
        self._value_to_value[val_id] = val

        cc = CongruenceClass(vn, val)
        cc.expression = expr
        self._vn_to_class[vn] = cc

        return vn

    def lookup(self, val: Value) -> Optional[int]:
        """Look up the value number for a value."""
        val_id = val.id if hasattr(val, 'id') else id(val)
        return self._value_to_vn.get(val_id)

    def get_vn(self, val: Value) -> int:
        """Get value number, creating one if needed."""
        val_id = val.id if hasattr(val, 'id') else id(val)
        vn = self._value_to_vn.get(val_id)
        if vn is not None:
            return vn
        # Unknown value: assign fresh number
        vn = self._next_vn
        self._next_vn += 1
        self._value_to_vn[val_id] = vn
        self._value_to_value[val_id] = val
        cc = CongruenceClass(vn, val)
        self._vn_to_class[vn] = cc
        return vn

    def get_class(self, vn: int) -> Optional[CongruenceClass]:
        return self._vn_to_class.get(vn)

    def get_leader(self, val: Value) -> Value:
        """Get the leader (canonical representative) for a value."""
        vn = self.lookup(val)
        if vn is not None:
            cc = self._vn_to_class.get(vn)
            if cc is not None:
                return cc.leader
        return val

    @property
    def num_value_numbers(self) -> int:
        return self._next_vn

    @property
    def num_congruence_classes(self) -> int:
        return len(self._vn_to_class)

    def clear(self) -> None:
        self._next_vn = 0
        self._expr_to_vn.clear()
        self._value_to_vn.clear()
        self._vn_to_class.clear()
        self._value_to_value.clear()

    def dump(self) -> str:
        lines = [f"ValueTable: {self._next_vn} value numbers"]
        for vn, cc in sorted(self._vn_to_class.items()):
            if cc.size > 1:
                lines.append(f"  VN{vn}: {cc.size} members, leader={cc.leader}")
        return "\n".join(lines)


# ─── Expression Builder ───────────────────────────────────────────────

class ExpressionBuilder:
    """Build ValueExpressions from instructions."""

    def __init__(self, vtable: ValueTable) -> None:
        self._vtable = vtable

    def build(self, inst: Instruction) -> ValueExpression:
        """Build a ValueExpression for an instruction."""
        if isinstance(inst, BinaryOp):
            return self._build_binary(inst)
        elif isinstance(inst, UnaryOp):
            return self._build_unary(inst)
        elif isinstance(inst, CompareOp):
            return self._build_compare(inst)
        elif isinstance(inst, CastInst):
            return self._build_cast(inst)
        elif isinstance(inst, SelectInst):
            return self._build_select(inst)
        elif isinstance(inst, PhiInst):
            return self._build_phi(inst)
        elif isinstance(inst, LoadInst):
            return self._build_load(inst)
        elif isinstance(inst, GetElementPtrInst):
            return self._build_gep(inst)
        elif isinstance(inst, CallInst):
            return self._build_call(inst)
        else:
            return ValueExpression(kind=ExprKind.UNKNOWN, extra=inst.id)

    def build_for_value(self, val: Value) -> ValueExpression:
        """Build expression for any value (constant, argument, instruction)."""
        if isinstance(val, Constant):
            return self._build_constant(val)
        elif isinstance(val, Argument):
            return self._build_argument(val)
        elif isinstance(val, Instruction):
            return self.build(val)
        return ValueExpression(kind=ExprKind.UNKNOWN, extra=id(val))

    def _build_constant(self, c: Constant) -> ValueExpression:
        return ValueExpression(
            kind=ExprKind.CONSTANT,
            extra=c.value if hasattr(c, 'value') else None,
            type_hash=hash(str(c.ir_type)) if hasattr(c, 'ir_type') else 0,
        )

    def _build_argument(self, arg: Argument) -> ValueExpression:
        return ValueExpression(
            kind=ExprKind.ARGUMENT,
            extra=arg.id,
            type_hash=hash(str(arg.ir_type)) if hasattr(arg, 'ir_type') else 0,
        )

    def _build_binary(self, inst: BinaryOp) -> ValueExpression:
        lvn = self._vtable.get_vn(inst.left)
        rvn = self._vtable.get_vn(inst.right)

        # Canonicalize commutative operations (smaller VN first)
        if self._is_commutative(inst.op) and lvn > rvn:
            lvn, rvn = rvn, lvn

        return ValueExpression(
            kind=ExprKind.BINARY,
            opcode=inst.op,
            operands=(lvn, rvn),
            type_hash=hash(str(inst.ir_type)),
        )

    def _build_unary(self, inst: UnaryOp) -> ValueExpression:
        ovn = self._vtable.get_vn(inst.operand)
        return ValueExpression(
            kind=ExprKind.UNARY,
            opcode=inst.op,
            operands=(ovn,),
            type_hash=hash(str(inst.ir_type)),
        )

    def _build_compare(self, inst: CompareOp) -> ValueExpression:
        lvn = self._vtable.get_vn(inst.left)
        rvn = self._vtable.get_vn(inst.right)

        # Canonicalize: swap operands and reverse predicate if needed
        pred = inst.predicate
        if lvn > rvn:
            lvn, rvn = rvn, lvn
            pred = self._swap_predicate(pred)

        return ValueExpression(
            kind=ExprKind.COMPARE,
            opcode=pred,
            operands=(lvn, rvn),
            type_hash=hash(str(inst.ir_type)),
        )

    def _build_cast(self, inst: CastInst) -> ValueExpression:
        ovn = self._vtable.get_vn(inst.operand)
        return ValueExpression(
            kind=ExprKind.CAST,
            opcode=inst.cast_kind,
            operands=(ovn,),
            type_hash=hash(str(inst.ir_type)),
        )

    def _build_select(self, inst: SelectInst) -> ValueExpression:
        cvn = self._vtable.get_vn(inst.condition)
        tvn = self._vtable.get_vn(inst.true_value)
        fvn = self._vtable.get_vn(inst.false_value)
        return ValueExpression(
            kind=ExprKind.SELECT,
            operands=(cvn, tvn, fvn),
            type_hash=hash(str(inst.ir_type)),
        )

    def _build_phi(self, inst: PhiInst) -> ValueExpression:
        # Sort incoming by block to canonicalize
        incoming_vns = tuple(sorted(
            (self._vtable.get_vn(val), block.id)
            for val, block in inst.incoming
        ))
        return ValueExpression(
            kind=ExprKind.PHI,
            operands=tuple(vn for vn, _ in incoming_vns),
            type_hash=hash(str(inst.ir_type)),
            extra=tuple(bid for _, bid in incoming_vns),
        )

    def _build_load(self, inst: LoadInst) -> ValueExpression:
        addr_vn = self._vtable.get_vn(inst.address)
        return ValueExpression(
            kind=ExprKind.LOAD,
            operands=(addr_vn,),
            type_hash=hash(str(inst.ir_type)),
            extra=getattr(inst, 'is_volatile', False),
        )

    def _build_gep(self, inst: GetElementPtrInst) -> ValueExpression:
        base_vn = self._vtable.get_vn(inst.base)
        idx_vns = tuple(self._vtable.get_vn(i) for i in inst.indices)
        return ValueExpression(
            kind=ExprKind.GEP,
            operands=(base_vn,) + idx_vns,
            type_hash=hash(str(inst.ir_type)),
        )

    def _build_call(self, inst: CallInst) -> ValueExpression:
        # Only number pure calls
        if not self._is_pure_call(inst):
            return ValueExpression(kind=ExprKind.UNKNOWN, extra=inst.id)
        callee_name = getattr(inst, 'callee_name', '') or (
            inst.callee.name if inst.callee and hasattr(inst.callee, 'name') else '')
        arg_vns = tuple(self._vtable.get_vn(a) for a in inst.arguments)
        return ValueExpression(
            kind=ExprKind.CALL,
            operands=arg_vns,
            type_hash=hash(str(inst.ir_type)),
            extra=callee_name,
        )

    def _is_commutative(self, op: BinOpKind) -> bool:
        return op in (BinOpKind.ADD, BinOpKind.MUL,
                     BinOpKind.AND, BinOpKind.OR, BinOpKind.XOR,
                     BinOpKind.FADD, BinOpKind.FMUL)

    def _swap_predicate(self, pred: CmpPredicate) -> CmpPredicate:
        swap_map = {
            CmpPredicate.EQ: CmpPredicate.EQ,
            CmpPredicate.NE: CmpPredicate.NE,
            CmpPredicate.SLT: CmpPredicate.SGT,
            CmpPredicate.SGT: CmpPredicate.SLT,
            CmpPredicate.SLE: CmpPredicate.SGE,
            CmpPredicate.SGE: CmpPredicate.SLE,
            CmpPredicate.ULT: CmpPredicate.UGT,
            CmpPredicate.UGT: CmpPredicate.ULT,
            CmpPredicate.ULE: CmpPredicate.UGE,
            CmpPredicate.UGE: CmpPredicate.ULE,
            CmpPredicate.OEQ: CmpPredicate.OEQ,
            CmpPredicate.ONE: CmpPredicate.ONE,
            CmpPredicate.OLT: CmpPredicate.OGT,
            CmpPredicate.OGT: CmpPredicate.OLT,
            CmpPredicate.OLE: CmpPredicate.OGE,
            CmpPredicate.OGE: CmpPredicate.OLE,
        }
        return swap_map.get(pred, pred)

    _PURE_FUNCTIONS = frozenset({
        "abs", "fabs", "sqrt", "sin", "cos", "tan",
        "exp", "log", "pow", "ceil", "floor", "round",
        "strlen", "memcmp",
    })

    def _is_pure_call(self, call: CallInst) -> bool:
        name = getattr(call, 'callee_name', '') or (
            call.callee.name if call.callee and hasattr(call.callee, 'name') else '')
        return name in self._PURE_FUNCTIONS


# ─── Memory Dependence Tracking ───────────────────────────────────────

class MemoryState:
    """Track memory state for load elimination in GVN.

    Maintains a mapping from address value number to the last
    stored value number, allowing loads to be eliminated when
    the stored value is still available.
    """

    def __init__(self) -> None:
        self._store_map: Dict[int, int] = {}  # addr_vn → stored_value_vn
        self._generation: int = 0

    def record_store(self, addr_vn: int, value_vn: int) -> None:
        self._store_map[addr_vn] = value_vn
        self._generation += 1

    def lookup_load(self, addr_vn: int) -> Optional[int]:
        return self._store_map.get(addr_vn)

    def invalidate_all(self) -> None:
        self._store_map.clear()
        self._generation += 1

    def invalidate_address(self, addr_vn: int) -> None:
        self._store_map.pop(addr_vn, None)

    def clone(self) -> "MemoryState":
        new = MemoryState()
        new._store_map = dict(self._store_map)
        new._generation = self._generation
        return new

    def merge(self, other: "MemoryState") -> "MemoryState":
        """Merge two memory states (at join points). Keep only common entries."""
        merged = MemoryState()
        for addr_vn in self._store_map:
            if addr_vn in other._store_map:
                if self._store_map[addr_vn] == other._store_map[addr_vn]:
                    merged._store_map[addr_vn] = self._store_map[addr_vn]
        return merged


# ─── GVN Pass ──────────────────────────────────────────────────────────

class GlobalValueNumbering(FunctionPass):
    """Global Value Numbering optimization pass.

    Uses hash-based value numbering to detect and eliminate
    redundant computations. Processes blocks in dominator tree
    order to ensure definitions dominate uses.

    Features:
    - Hash consing for efficient expression comparison
    - Commutative operation canonicalization
    - Load elimination via memory state tracking
    - Phi node simplification
    """

    _name = "gvn"
    _description = "Global value numbering for redundant computation elimination"
    _required_analyses = ["domtree"]
    _invalidated_analyses = ["cfg", "domtree"]

    def __init__(self) -> None:
        super().__init__()
        self._vtable = ValueTable()
        self._expr_builder: Optional[ExpressionBuilder] = None
        self._eliminated: int = 0

    def run_on_function(self, function: Function, analyses: AnalysisManager) -> PassResult:
        self._vtable.clear()
        self._expr_builder = ExpressionBuilder(self._vtable)
        self._eliminated = 0

        # Number arguments
        for arg in function.arguments:
            expr = self._expr_builder.build_for_value(arg)
            self._vtable.lookup_or_add(arg, expr)

        # Process blocks in RPO (approximation of dom tree preorder)
        rpo = function.iter_blocks_rpo()
        mem_states: Dict[int, MemoryState] = {}

        for block in rpo:
            mem = self._compute_incoming_memory(block, mem_states)
            mem = self._process_block(block, mem)
            mem_states[block.id] = mem

        if self._eliminated > 0:
            self.stats.instructions_removed += self._eliminated
            self.stats.increment("gvn_eliminated", self._eliminated)
            logger.debug(f"GVN: eliminated {self._eliminated} redundant instructions in {function.name}")

        return PassResult.CHANGED if self._eliminated > 0 else PassResult.UNCHANGED

    def _compute_incoming_memory(self, block: BasicBlock,
                                  mem_states: Dict[int, MemoryState]) -> MemoryState:
        """Compute the incoming memory state for a block."""
        preds = block.predecessors
        if not preds:
            return MemoryState()

        pred_states = [mem_states.get(p.id) for p in preds]
        pred_states = [s for s in pred_states if s is not None]

        if not pred_states:
            return MemoryState()
        if len(pred_states) == 1:
            return pred_states[0].clone()

        # Merge states from multiple predecessors
        result = pred_states[0]
        for state in pred_states[1:]:
            result = result.merge(state)
        return result

    def _process_block(self, block: BasicBlock, mem: MemoryState) -> MemoryState:
        """Process all instructions in a block, performing GVN."""
        to_remove: List[Instruction] = []

        for inst in list(block.instructions):
            if isinstance(inst, StoreInst):
                self._process_store(inst, mem)
                continue

            if isinstance(inst, CallInst) and not self._expr_builder._is_pure_call(inst):
                # Non-pure call invalidates all memory
                mem.invalidate_all()
                self._vtable.get_vn(inst)
                continue

            if isinstance(inst, (ReturnInst, BranchInst)):
                continue

            # Try load elimination
            if isinstance(inst, LoadInst):
                eliminated = self._try_eliminate_load(inst, mem)
                if eliminated:
                    to_remove.append(inst)
                    continue

            # Build expression and look up value number
            expr = self._expr_builder.build(inst)
            existing_vn = self._vtable._expr_to_vn.get(expr)

            if existing_vn is not None:
                # This expression was seen before — try to replace
                cc = self._vtable.get_class(existing_vn)
                if cc is not None and cc.leader is not inst:
                    leader = cc.leader
                    # Verify leader dominates inst
                    if self._can_replace(inst, leader):
                        self._replace_uses(inst, leader)
                        to_remove.append(inst)
                        self._eliminated += 1
                        cc.add_member(inst.id)
                        self._vtable._value_to_vn[inst.id] = existing_vn
                        continue

            # New expression or non-replaceable: add to table
            self._vtable.lookup_or_add(inst, expr)

            # Record load in memory state
            if isinstance(inst, LoadInst):
                addr_vn = self._vtable.get_vn(inst.address)
                inst_vn = self._vtable.get_vn(inst)
                mem.record_store(addr_vn, inst_vn)

        for inst in to_remove:
            try:
                block.remove(inst)
            except ValueError:
                pass

        return mem

    def _process_store(self, store: StoreInst, mem: MemoryState) -> None:
        """Process a store instruction for memory tracking."""
        addr_vn = self._vtable.get_vn(store.address)
        val_vn = self._vtable.get_vn(store.value)
        mem.record_store(addr_vn, val_vn)

    def _try_eliminate_load(self, load: LoadInst, mem: MemoryState) -> bool:
        """Try to eliminate a load using stored value."""
        if getattr(load, 'is_volatile', False):
            return False

        addr_vn = self._vtable.get_vn(load.address)
        stored_vn = mem.lookup_load(addr_vn)

        if stored_vn is None:
            return False

        # Find a value with the stored VN
        cc = self._vtable.get_class(stored_vn)
        if cc is None:
            return False

        leader = cc.leader
        if leader is load:
            return False

        # Type check
        if (hasattr(leader, 'ir_type') and hasattr(load, 'ir_type') and
                str(leader.ir_type) != str(load.ir_type)):
            return False

        self._replace_uses(load, leader)
        self._eliminated += 1
        return True

    def _can_replace(self, inst: Instruction, leader: Value) -> bool:
        """Check if leader can replace inst (must dominate)."""
        if isinstance(leader, (Constant, Argument)):
            return True
        if not isinstance(leader, Instruction):
            return False
        if leader.parent is None or inst.parent is None:
            return False
        # Simple dominance check: if in same block, leader must come first
        if leader.parent is inst.parent:
            for i in leader.parent.instructions:
                if i is leader:
                    return True
                if i is inst:
                    return False
        # Different blocks: check block dominance
        return self._block_dominates(leader.parent, inst.parent)

    def _block_dominates(self, a: BasicBlock, b: BasicBlock) -> bool:
        """Check if block a dominates block b."""
        current = b
        visited: Set[int] = set()
        while current is not None:
            if current is a:
                return True
            if current.id in visited:
                break
            visited.add(current.id)
            current = current.idom
        return False

    def _replace_uses(self, old: Value, new: Value) -> None:
        """Replace all uses of old with new."""
        if not hasattr(old, 'users'):
            return
        for user in list(old.users):
            if isinstance(user, Instruction):
                self._substitute(user, old, new)
                if hasattr(new, 'users') and not isinstance(new, Constant):
                    new.users.append(user)

    def _substitute(self, user: Instruction, old: Value, new: Value) -> None:
        if isinstance(user, BinaryOp):
            if user.left is old:
                user.left = new
            if user.right is old:
                user.right = new
        elif isinstance(user, UnaryOp):
            if user.operand is old:
                user.operand = new
        elif isinstance(user, CompareOp):
            if user.left is old:
                user.left = new
            if user.right is old:
                user.right = new
        elif isinstance(user, CastInst):
            if user.operand is old:
                user.operand = new
        elif isinstance(user, SelectInst):
            if user.condition is old:
                user.condition = new
            if user.true_value is old:
                user.true_value = new
            if user.false_value is old:
                user.false_value = new
        elif isinstance(user, PhiInst):
            user.incoming = [(new if v is old else v, b) for v, b in user.incoming]
        elif isinstance(user, BranchInst) and user.is_conditional:
            if user.condition is old:
                user.condition = new
        elif isinstance(user, ReturnInst):
            if user.value is old:
                user.value = new
        elif isinstance(user, StoreInst):
            if user.value is old:
                user.value = new
            if user.address is old:
                user.address = new
        elif isinstance(user, LoadInst):
            if user.address is old:
                user.address = new
        elif isinstance(user, CallInst):
            user.arguments = [new if a is old else a for a in user.arguments]
        elif isinstance(user, GetElementPtrInst):
            if user.base is old:
                user.base = new
            user.indices = [new if i is old else i for i in user.indices]
