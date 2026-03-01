"""
Symbolic execution state management.

Provides SymbolicState with symbolic variables (z3 expressions),
path constraints, memory model (symbolic arrays), and call stack.
Supports state forking, merging, and comparison.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Tuple, Set, Any, Iterator

import z3

from ..ir.types import (
    IRType, IntType, FloatType, PointerType, VoidType,
    ArrayType, StructType, Signedness, FloatKind,
)
from ..ir.instructions import Value, Constant, Argument
from ..ir.basic_block import BasicBlock


# ---------------------------------------------------------------------------
# Symbolic value wrapper
# ---------------------------------------------------------------------------

@dataclass
class SymbolicValue:
    """A symbolic value backed by a Z3 expression."""
    name: str
    z3_expr: z3.ExprRef
    ir_type: Optional[IRType] = None
    provenance: Optional[str] = None  # Pointer provenance tag

    @property
    def sort(self) -> z3.SortRef:
        return self.z3_expr.sort()

    @property
    def is_bitvector(self) -> bool:
        return z3.is_bv(self.z3_expr)

    @property
    def is_fp(self) -> bool:
        return z3.is_fp(self.z3_expr)

    @property
    def is_bool(self) -> bool:
        return z3.is_bool(self.z3_expr)

    @property
    def is_concrete(self) -> bool:
        return z3.is_bv_value(self.z3_expr) or z3.is_fp_value(self.z3_expr)

    def __repr__(self) -> str:
        return f"SVal({self.name}: {self.z3_expr})"


# ---------------------------------------------------------------------------
# Path constraint
# ---------------------------------------------------------------------------

@dataclass
class PathConstraint:
    """A constraint on the current execution path."""
    condition: z3.BoolRef
    is_branch_taken: bool
    source_block: Optional[str] = None
    description: str = ""

    def __repr__(self) -> str:
        taken = "T" if self.is_branch_taken else "F"
        return f"PC({taken}: {self.condition})"


# ---------------------------------------------------------------------------
# Call frame
# ---------------------------------------------------------------------------

@dataclass
class CallFrame:
    """A frame on the symbolic call stack."""
    function_name: str
    return_block: Optional[str] = None
    return_value_name: Optional[str] = None
    local_vars: Dict[str, SymbolicValue] = field(default_factory=dict)
    depth: int = 0

    def __repr__(self) -> str:
        return f"Frame({self.function_name}, depth={self.depth})"


# ---------------------------------------------------------------------------
# Symbolic memory
# ---------------------------------------------------------------------------

class SymbolicMemory:
    """
    Symbolic memory using Z3 arrays.
    
    Models memory as a mapping from bitvector addresses to bitvector values.
    Supports symbolic load/store with aliasing and fresh symbol allocation.
    """

    def __init__(self, address_width: int = 64, byte_width: int = 8):
        self.address_width = address_width
        self.byte_width = byte_width
        self._addr_sort = z3.BitVecSort(address_width)
        self._byte_sort = z3.BitVecSort(byte_width)

        # Main memory as Z3 array: addr -> byte
        self._mem_version = 0
        self._memory = z3.Array(
            f"mem_{self._mem_version}",
            self._addr_sort,
            self._byte_sort,
        )

        # Allocation tracking
        self._allocations: Dict[str, Tuple[z3.BitVecRef, int]] = {}
        self._next_alloc_id = 0
        self._freed: Set[str] = set()

        # Operation log for counterexample generation
        self._op_log: List[Dict[str, Any]] = []

        # Fresh symbol counter
        self._symbol_counter = 0

    def store(
        self,
        address: z3.BitVecRef,
        value: z3.BitVecRef,
        num_bytes: int,
    ) -> None:
        """Store a value at a symbolic address (little-endian byte order)."""
        for i in range(num_bytes):
            byte_val = z3.Extract(
                (i + 1) * self.byte_width - 1,
                i * self.byte_width,
                value,
            ) if num_bytes > 1 else value

            offset = z3.BitVecVal(i, self.address_width)
            addr_i = address + offset

            self._mem_version += 1
            self._memory = z3.Store(self._memory, addr_i, byte_val)

        self._op_log.append({
            "type": "store",
            "address": address,
            "value": value,
            "bytes": num_bytes,
        })

    def load(
        self,
        address: z3.BitVecRef,
        num_bytes: int,
    ) -> z3.BitVecRef:
        """Load a value from a symbolic address (little-endian byte order)."""
        if num_bytes == 1:
            result = z3.Select(self._memory, address)
        else:
            bytes_list = []
            for i in range(num_bytes):
                offset = z3.BitVecVal(i, self.address_width)
                addr_i = address + offset
                byte_val = z3.Select(self._memory, addr_i)
                bytes_list.append(byte_val)

            # Little-endian: first byte is least significant
            result = bytes_list[num_bytes - 1]
            for i in range(num_bytes - 2, -1, -1):
                result = z3.Concat(result, bytes_list[i])

        self._op_log.append({
            "type": "load",
            "address": address,
            "bytes": num_bytes,
        })

        return result

    def allocate(
        self,
        size: int,
        name: str = "",
        alignment: int = 1,
    ) -> Tuple[str, z3.BitVecRef]:
        """Allocate a symbolic memory region. Returns (alloc_id, base_address)."""
        alloc_id = f"alloc_{self._next_alloc_id}"
        self._next_alloc_id += 1

        if name:
            alloc_id = f"{alloc_id}_{name}"

        base = z3.BitVec(f"addr_{alloc_id}", self.address_width)
        self._allocations[alloc_id] = (base, size)

        self._op_log.append({
            "type": "alloc",
            "id": alloc_id,
            "size": size,
            "address": base,
        })

        return alloc_id, base

    def free(self, alloc_id: str) -> bool:
        """Free an allocation. Returns False if double-free."""
        if alloc_id in self._freed:
            return False
        if alloc_id not in self._allocations:
            return False
        self._freed.add(alloc_id)

        self._op_log.append({
            "type": "free",
            "id": alloc_id,
        })
        return True

    def is_allocated(self, alloc_id: str) -> bool:
        return alloc_id in self._allocations and alloc_id not in self._freed

    def fresh_symbol(self, name: str, width: int) -> z3.BitVecRef:
        """Create a fresh symbolic bitvector."""
        self._symbol_counter += 1
        return z3.BitVec(f"{name}_{self._symbol_counter}", width)

    def fresh_fp_symbol(self, name: str, ebits: int = 11, sbits: int = 53) -> z3.FPRef:
        """Create a fresh symbolic floating-point value."""
        self._symbol_counter += 1
        return z3.FP(f"{name}_{self._symbol_counter}", z3.FPSort(ebits, sbits))

    def copy(self) -> SymbolicMemory:
        """Create a deep copy of this memory state."""
        new_mem = SymbolicMemory(self.address_width, self.byte_width)
        new_mem._memory = self._memory
        new_mem._mem_version = self._mem_version
        new_mem._allocations = dict(self._allocations)
        new_mem._freed = set(self._freed)
        new_mem._next_alloc_id = self._next_alloc_id
        new_mem._symbol_counter = self._symbol_counter
        new_mem._op_log = list(self._op_log)
        return new_mem

    def get_op_log(self) -> List[Dict[str, Any]]:
        return list(self._op_log)

    def allocation_constraints(self) -> List[z3.BoolRef]:
        """Generate constraints ensuring allocations don't overlap and are non-null."""
        constraints: List[z3.BoolRef] = []
        allocs = [(aid, base, size) for aid, (base, size) in self._allocations.items()
                  if aid not in self._freed]

        for aid, base, size in allocs:
            # Non-null constraint
            constraints.append(base != z3.BitVecVal(0, self.address_width))
            # Positive address
            constraints.append(z3.UGT(base, z3.BitVecVal(0x1000, self.address_width)))

        # Non-overlap constraints
        for i in range(len(allocs)):
            for j in range(i + 1, len(allocs)):
                _, base_i, size_i = allocs[i]
                _, base_j, size_j = allocs[j]
                # Either i ends before j starts, or j ends before i starts
                constraints.append(z3.Or(
                    z3.UGE(base_j, base_i + z3.BitVecVal(size_i, self.address_width)),
                    z3.UGE(base_i, base_j + z3.BitVecVal(size_j, self.address_width)),
                ))

        return constraints


# ---------------------------------------------------------------------------
# Symbolic state
# ---------------------------------------------------------------------------

class SymbolicState:
    """
    Complete symbolic execution state.
    
    Contains symbolic variables, path constraints, memory model,
    and call stack. Supports forking and merging.
    """

    def __init__(
        self,
        function_name: str = "",
        address_width: int = 64,
    ):
        self.function_name = function_name
        self.address_width = address_width

        # Symbolic variable bindings: SSA name → SymbolicValue
        self._variables: Dict[str, SymbolicValue] = {}

        # Path constraints
        self._path_constraints: List[PathConstraint] = []

        # Memory
        self.memory = SymbolicMemory(address_width)

        # Call stack
        self._call_stack: List[CallFrame] = [
            CallFrame(function_name=function_name, depth=0)
        ]

        # Current location
        self.current_block: Optional[str] = None
        self.current_inst_index: int = 0

        # Execution status
        self.is_terminated: bool = False
        self.termination_reason: str = ""
        self.return_value: Optional[SymbolicValue] = None

        # Path identifier
        self._path_id: int = 0
        self._fork_count: int = 0

        # Fresh variable counter
        self._fresh_counter: int = 0

        # Block visit counts (for loop detection)
        self._block_visits: Dict[str, int] = {}

        # Covered blocks for coverage tracking
        self.covered_blocks: Set[str] = set()

    # -- Variable management --

    def get_var(self, name: str) -> Optional[SymbolicValue]:
        """Get a symbolic variable by name."""
        sv = self._variables.get(name)
        if sv is not None:
            return sv
        # Check call stack local vars
        if self._call_stack:
            return self._call_stack[-1].local_vars.get(name)
        return None

    def set_var(self, name: str, value: SymbolicValue) -> None:
        """Set a symbolic variable."""
        self._variables[name] = value

    def set_z3_var(self, name: str, expr: z3.ExprRef, ir_type: Optional[IRType] = None) -> None:
        """Convenience: set a variable from a Z3 expression."""
        self._variables[name] = SymbolicValue(name=name, z3_expr=expr, ir_type=ir_type)

    def has_var(self, name: str) -> bool:
        return name in self._variables

    def all_variables(self) -> Dict[str, SymbolicValue]:
        return dict(self._variables)

    # -- Path constraints --

    def add_constraint(
        self,
        condition: z3.BoolRef,
        is_branch_taken: bool = True,
        source_block: Optional[str] = None,
        description: str = "",
    ) -> None:
        """Add a path constraint."""
        self._path_constraints.append(PathConstraint(
            condition=condition,
            is_branch_taken=is_branch_taken,
            source_block=source_block,
            description=description,
        ))

    @property
    def path_constraints(self) -> List[PathConstraint]:
        return list(self._path_constraints)

    @property
    def constraint_expressions(self) -> List[z3.BoolRef]:
        return [pc.condition for pc in self._path_constraints]

    @property
    def path_condition(self) -> z3.BoolRef:
        """Get the conjunction of all path constraints."""
        exprs = self.constraint_expressions
        if not exprs:
            return z3.BoolVal(True)
        if len(exprs) == 1:
            return exprs[0]
        return z3.And(*exprs)

    @property
    def path_length(self) -> int:
        return len(self._path_constraints)

    # -- Call stack --

    def push_frame(self, function_name: str, return_block: Optional[str] = None,
                   return_var: Optional[str] = None) -> None:
        """Push a new call frame."""
        depth = len(self._call_stack)
        self._call_stack.append(CallFrame(
            function_name=function_name,
            return_block=return_block,
            return_value_name=return_var,
            depth=depth,
        ))

    def pop_frame(self) -> Optional[CallFrame]:
        """Pop the top call frame."""
        if len(self._call_stack) <= 1:
            return None
        return self._call_stack.pop()

    @property
    def call_depth(self) -> int:
        return len(self._call_stack)

    @property
    def current_frame(self) -> CallFrame:
        return self._call_stack[-1]

    # -- Block tracking --

    def visit_block(self, block_name: str) -> int:
        """Record a block visit. Returns visit count."""
        count = self._block_visits.get(block_name, 0) + 1
        self._block_visits[block_name] = count
        self.covered_blocks.add(block_name)
        self.current_block = block_name
        self.current_inst_index = 0
        return count

    def block_visit_count(self, block_name: str) -> int:
        return self._block_visits.get(block_name, 0)

    # -- State forking and merging --

    def fork(self) -> SymbolicState:
        """Create a forked copy of this state (for branching)."""
        new_state = SymbolicState(
            function_name=self.function_name,
            address_width=self.address_width,
        )
        new_state._variables = dict(self._variables)
        new_state._path_constraints = list(self._path_constraints)
        new_state.memory = self.memory.copy()
        new_state._call_stack = [
            CallFrame(
                function_name=f.function_name,
                return_block=f.return_block,
                return_value_name=f.return_value_name,
                local_vars=dict(f.local_vars),
                depth=f.depth,
            )
            for f in self._call_stack
        ]
        new_state.current_block = self.current_block
        new_state.current_inst_index = self.current_inst_index
        new_state.is_terminated = self.is_terminated
        new_state.termination_reason = self.termination_reason
        new_state.return_value = self.return_value
        new_state._path_id = self._path_id
        new_state._fork_count = self._fork_count + 1
        new_state._fresh_counter = self._fresh_counter
        new_state._block_visits = dict(self._block_visits)
        new_state.covered_blocks = set(self.covered_blocks)
        return new_state

    @staticmethod
    def merge(states: List[SymbolicState]) -> Optional[SymbolicState]:
        """
        Merge multiple states at a join point.
        
        Variables get if-then-else expressions based on path conditions.
        """
        if not states:
            return None
        if len(states) == 1:
            return states[0]

        base = states[0].fork()

        # Merge variables using ITE on path conditions
        all_var_names: Set[str] = set()
        for s in states:
            all_var_names.update(s._variables.keys())

        for var_name in all_var_names:
            # Build ITE chain
            values = []
            for s in states:
                sv = s.get_var(var_name)
                if sv is not None:
                    values.append((s.path_condition, sv.z3_expr))

            if not values:
                continue

            if len(values) == 1:
                base.set_z3_var(var_name, values[0][1])
            else:
                # Chain: ITE(cond1, val1, ITE(cond2, val2, ...))
                merged_expr = values[-1][1]
                for i in range(len(values) - 2, -1, -1):
                    cond, val = values[i]
                    merged_expr = z3.If(cond, val, merged_expr)
                base.set_z3_var(var_name, merged_expr)

        # Merge path constraints: disjunction of all paths
        if len(states) > 1:
            path_conds = [s.path_condition for s in states]
            base._path_constraints = [PathConstraint(
                condition=z3.Or(*path_conds),
                is_branch_taken=True,
                description="merged path condition",
            )]

        # Merge coverage
        for s in states:
            base.covered_blocks.update(s.covered_blocks)

        return base

    # -- Fresh symbols --

    def fresh_bv(self, name: str, width: int) -> z3.BitVecRef:
        """Create a fresh symbolic bitvector."""
        self._fresh_counter += 1
        return z3.BitVec(f"{name}_{self._fresh_counter}", width)

    def fresh_fp(self, name: str, sort: z3.FPSortRef) -> z3.FPRef:
        """Create a fresh symbolic float."""
        self._fresh_counter += 1
        return z3.FP(f"{name}_{self._fresh_counter}", sort)

    def fresh_bool(self, name: str) -> z3.BoolRef:
        """Create a fresh symbolic boolean."""
        self._fresh_counter += 1
        return z3.Bool(f"{name}_{self._fresh_counter}")

    # -- Termination --

    def terminate(self, reason: str, return_value: Optional[SymbolicValue] = None) -> None:
        """Mark this state as terminated."""
        self.is_terminated = True
        self.termination_reason = reason
        self.return_value = return_value

    # -- Utility --

    def __repr__(self) -> str:
        return (
            f"SymbolicState(func={self.function_name}, "
            f"block={self.current_block}, "
            f"vars={len(self._variables)}, "
            f"constraints={self.path_length}, "
            f"depth={self.call_depth})"
        )

    def summary(self) -> str:
        lines = [
            f"State: {self.function_name}",
            f"  Block: {self.current_block}",
            f"  Variables: {len(self._variables)}",
            f"  Path constraints: {self.path_length}",
            f"  Call depth: {self.call_depth}",
            f"  Covered blocks: {len(self.covered_blocks)}",
            f"  Terminated: {self.is_terminated} ({self.termination_reason})",
        ]
        return "\n".join(lines)
