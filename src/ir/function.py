"""
Function representation for the Cross-Language Equivalence Verifier IR.

An SSA function containing parameters, basic blocks, a control-flow graph,
local variable table, SSA validation, and pretty-printing.
"""

from __future__ import annotations

from collections import deque
from typing import Iterator, Optional, Sequence

from .basic_block import BasicBlock
from .instructions import (
    Argument,
    BranchInst,
    Instruction,
    PhiInst,
    ReturnInst,
    SwitchInst,
    Value,
    is_terminator,
    get_successors,
)
from .types import FunctionType, IRType, VoidType


class Function:
    """An SSA function.

    Attributes:
        name: function name / symbol.
        func_type: the FunctionType describing signature.
        arguments: parameter Value objects.
        linkage: linkage kind (external, internal, private, …).
        language: source language tag ("c" or "rust").
    """
    __slots__ = (
        "name", "func_type", "_arguments", "_blocks",
        "_local_vars", "linkage", "language", "attributes",
        "_value_names",
    )

    def __init__(
        self,
        name: str,
        func_type: FunctionType,
        linkage: str = "external",
        language: str = "",
    ) -> None:
        self.name = name
        self.func_type = func_type
        self.linkage = linkage
        self.language = language
        self.attributes: dict[str, str] = {}

        # Create argument values
        self._arguments: list[Argument] = []
        for i, pt in enumerate(func_type.param_types):
            self._arguments.append(Argument(pt, i))

        self._blocks: list[BasicBlock] = []
        self._local_vars: dict[str, Value] = {}
        self._value_names: dict[str, Value] = {}

    # ------------------------------------------------------------------
    # Arguments
    # ------------------------------------------------------------------

    @property
    def arguments(self) -> list[Argument]:
        return list(self._arguments)

    @property
    def num_arguments(self) -> int:
        return len(self._arguments)

    def get_argument(self, index: int) -> Argument:
        return self._arguments[index]

    def set_argument_name(self, index: int, name: str) -> None:
        self._arguments[index].name = name

    @property
    def return_type(self) -> IRType:
        return self.func_type.return_type

    @property
    def is_void_return(self) -> bool:
        return isinstance(self.return_type, VoidType)

    @property
    def is_variadic(self) -> bool:
        return self.func_type.is_variadic

    # ------------------------------------------------------------------
    # Basic blocks
    # ------------------------------------------------------------------

    @property
    def blocks(self) -> list[BasicBlock]:
        return list(self._blocks)

    @property
    def num_blocks(self) -> int:
        return len(self._blocks)

    @property
    def entry_block(self) -> BasicBlock | None:
        return self._blocks[0] if self._blocks else None

    @property
    def exit_blocks(self) -> list[BasicBlock]:
        """Return all blocks ending with a return instruction."""
        return [b for b in self._blocks if b.is_exit]

    def add_block(self, block: BasicBlock) -> BasicBlock:
        """Append a block to the function."""
        block.parent = self
        self._blocks.append(block)
        return block

    def create_block(self, name: str = "") -> BasicBlock:
        """Create a new block and add it to the function."""
        block = BasicBlock(name=name, parent=self)
        self._blocks.append(block)
        return block

    def insert_block_after(self, after: BasicBlock, block: BasicBlock) -> None:
        idx = self._blocks.index(after)
        block.parent = self
        self._blocks.insert(idx + 1, block)

    def insert_block_before(self, before: BasicBlock, block: BasicBlock) -> None:
        idx = self._blocks.index(before)
        block.parent = self
        self._blocks.insert(idx, block)

    def remove_block(self, block: BasicBlock) -> None:
        """Remove a block from the function and clean up edges."""
        # Remove as predecessor from successors
        for succ in block.successors:
            succ.remove_predecessor(block)
        # Remove as successor from predecessors
        for pred in block.predecessors:
            pred._successors = [s for s in pred._successors if s is not block]
        block.parent = None
        self._blocks.remove(block)

    def get_block_by_name(self, name: str) -> BasicBlock | None:
        for b in self._blocks:
            if b.name == name:
                return b
        return None

    # ------------------------------------------------------------------
    # Local variable table
    # ------------------------------------------------------------------

    def add_local(self, name: str, value: Value) -> None:
        self._local_vars[name] = value

    def get_local(self, name: str) -> Value | None:
        return self._local_vars.get(name)

    @property
    def locals(self) -> dict[str, Value]:
        return dict(self._local_vars)

    # ------------------------------------------------------------------
    # Value naming
    # ------------------------------------------------------------------

    def register_value(self, name: str, value: Value) -> None:
        """Register a named value for lookup."""
        self._value_names[name] = value
        value.name = name

    def lookup_value(self, name: str) -> Value | None:
        return self._value_names.get(name)

    # ------------------------------------------------------------------
    # CFG construction & traversal
    # ------------------------------------------------------------------

    def cfg_predecessors(self, block: BasicBlock) -> list[BasicBlock]:
        return block.predecessors

    def cfg_successors(self, block: BasicBlock) -> list[BasicBlock]:
        return block.successors

    def iter_blocks_rpo(self) -> list[BasicBlock]:
        """Return blocks in reverse post-order (useful for dataflow)."""
        if not self._blocks:
            return []
        visited: set[int] = set()
        rpo: list[BasicBlock] = []

        def dfs(block: BasicBlock) -> None:
            visited.add(block.id)
            for succ in block.successors:
                if succ.id not in visited:
                    dfs(succ)
            rpo.append(block)

        dfs(self._blocks[0])
        rpo.reverse()
        return rpo

    def iter_blocks_bfs(self) -> list[BasicBlock]:
        """Return blocks in BFS order from the entry."""
        if not self._blocks:
            return []
        visited: set[int] = set()
        result: list[BasicBlock] = []
        queue: deque[BasicBlock] = deque([self._blocks[0]])
        visited.add(self._blocks[0].id)
        while queue:
            block = queue.popleft()
            result.append(block)
            for succ in block.successors:
                if succ.id not in visited:
                    visited.add(succ.id)
                    queue.append(succ)
        return result

    def compute_dominators(self) -> None:
        """Compute the dominator tree using the iterative algorithm.

        Sets the idom, dom_children fields on each BasicBlock.
        """
        if not self._blocks:
            return

        rpo = self.iter_blocks_rpo()
        block_to_idx = {b.id: i for i, b in enumerate(rpo)}
        entry = rpo[0]
        n = len(rpo)

        # Initialize idoms: entry dominates itself, others undefined
        idoms: list[int] = [-1] * n
        idoms[0] = 0

        def intersect(b1: int, b2: int) -> int:
            finger1, finger2 = b1, b2
            while finger1 != finger2:
                while finger1 > finger2:
                    finger1 = idoms[finger1]
                while finger2 > finger1:
                    finger2 = idoms[finger2]
            return finger1

        changed = True
        while changed:
            changed = False
            for i in range(1, n):
                block = rpo[i]
                # Pick the first processed predecessor
                new_idom = -1
                for pred in block.predecessors:
                    pred_idx = block_to_idx.get(pred.id, -1)
                    if pred_idx == -1:
                        continue
                    if idoms[pred_idx] == -1:
                        continue
                    if new_idom == -1:
                        new_idom = pred_idx
                    else:
                        new_idom = intersect(new_idom, pred_idx)
                if new_idom != -1 and idoms[i] != new_idom:
                    idoms[i] = new_idom
                    changed = True

        # Set idom on blocks
        for i, block in enumerate(rpo):
            block._dom_children = []
            if idoms[i] == i:
                block.idom = None  # entry block
            else:
                block.idom = rpo[idoms[i]]

        # Build dom_children
        for block in rpo:
            if block.idom is not None:
                block.idom._dom_children.append(block)

        # Compute dominance frontiers
        for block in rpo:
            block.dominance_frontier = []

        for block in rpo:
            if len(block.predecessors) < 2:
                continue
            for pred in block.predecessors:
                runner: BasicBlock | None = pred
                while runner is not None and runner is not block.idom:
                    if block not in runner.dominance_frontier:
                        runner.dominance_frontier.append(block)
                    runner = runner.idom

    # ------------------------------------------------------------------
    # Instruction iteration
    # ------------------------------------------------------------------

    def iter_instructions(self) -> Iterator[Instruction]:
        """Iterate over all instructions in all blocks."""
        for block in self._blocks:
            yield from block

    @property
    def instruction_count(self) -> int:
        return sum(len(b) for b in self._blocks)

    def all_values(self) -> list[Value]:
        """Return all values defined in this function (args + instructions)."""
        values: list[Value] = list(self._arguments)
        for inst in self.iter_instructions():
            values.append(inst)
        return values

    # ------------------------------------------------------------------
    # SSA validation
    # ------------------------------------------------------------------

    def validate_ssa(self) -> list[str]:
        """Validate SSA properties of this function.

        Checks:
        - Every use is dominated by its definition (or is an argument).
        - Phi nodes have correct incoming blocks.
        - No duplicate definitions.
        """
        errors: list[str] = []
        self.compute_dominators()

        # Build a map from Value id to defining block
        def_block: dict[int, BasicBlock] = {}
        for block in self._blocks:
            for inst in block:
                if inst.id in def_block:
                    errors.append(f"Duplicate definition: {inst.display_name}")
                def_block[inst.id] = block

        # Arguments are defined in the entry block (conceptually)
        entry = self.entry_block
        for arg in self._arguments:
            if entry is not None:
                def_block[arg.id] = entry

        # Check dominance for each use
        for block in self._blocks:
            for inst in block:
                for op in inst.operands:
                    if isinstance(op, Argument):
                        continue
                    op_def_block = def_block.get(op.id)
                    if op_def_block is None:
                        # Could be a constant or external
                        continue
                    if isinstance(inst, PhiInst):
                        # For phi nodes, the value must dominate the
                        # incoming edge's source block (not this block)
                        for v, pred in inst.incoming:
                            if v is op:
                                if not op_def_block.dominates(pred):
                                    errors.append(
                                        f"SSA violation: {op.display_name} defined in "
                                        f"'{op_def_block.name}' does not dominate phi "
                                        f"predecessor '{pred.name}' in '{block.name}'"
                                    )
                    else:
                        if not op_def_block.dominates(block):
                            errors.append(
                                f"SSA violation: {op.display_name} defined in "
                                f"'{op_def_block.name}' does not dominate use in "
                                f"'{block.name}'"
                            )
                        elif op_def_block is block:
                            # Same block: definition must come before use
                            def_idx = None
                            use_idx = None
                            for i, bi in enumerate(block):
                                if bi is op:
                                    def_idx = i
                                if bi is inst:
                                    use_idx = i
                            if def_idx is not None and use_idx is not None:
                                if def_idx >= use_idx:
                                    errors.append(
                                        f"SSA violation: {op.display_name} used before "
                                        f"definition in '{block.name}'"
                                    )

        return errors

    def validate(self) -> list[str]:
        """Full structural + SSA validation."""
        errors: list[str] = []

        if not self._blocks:
            errors.append(f"Function '{self.name}' has no basic blocks")
            return errors

        # Validate each block
        for block in self._blocks:
            block_errors = block.validate()
            for e in block_errors:
                errors.append(f"Function '{self.name}': {e}")

        # Entry block should have no predecessors
        entry = self.entry_block
        if entry and entry.num_predecessors > 0:
            errors.append(f"Function '{self.name}': entry block has predecessors")

        # Check return types
        for block in self.exit_blocks:
            ret = block.terminator
            if isinstance(ret, ReturnInst):
                if ret.is_void_return:
                    if not self.is_void_return:
                        errors.append(
                            f"Function '{self.name}': void return in non-void function"
                        )
                else:
                    ret_val = ret.return_value
                    if ret_val is not None and ret_val.type != self.return_type:
                        errors.append(
                            f"Function '{self.name}': return type mismatch: "
                            f"expected {self.return_type}, got {ret_val.type}"
                        )

        # Check for unreachable blocks
        reachable = set(b.id for b in self.iter_blocks_bfs())
        for block in self._blocks:
            if block.id not in reachable:
                errors.append(f"Function '{self.name}': unreachable block '{block.name}'")

        # SSA validation
        errors.extend(self.validate_ssa())

        return errors

    # ------------------------------------------------------------------
    # Pretty printing
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"Function({self.name}, {self.num_blocks} blocks)"

    def dump(self) -> str:
        """Return a multi-line textual representation."""
        lines: list[str] = []
        # Signature
        params = ", ".join(
            f"{a.display_name}: {a.type}" for a in self._arguments
        )
        ret = str(self.return_type)
        linkage = f"{self.linkage} " if self.linkage else ""
        lines.append(f"define {linkage}{ret} @{self.name}({params}) {{")

        for block in self._blocks:
            lines.append(block.dump())

        lines.append("}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Clone
    # ------------------------------------------------------------------

    def clone(self, new_name: str | None = None) -> "Function":
        """Create a deep copy of this function with fresh block/value IDs."""
        new_func = Function(
            name=new_name or f"{self.name}.clone",
            func_type=self.func_type,
            linkage=self.linkage,
            language=self.language,
        )
        new_func.attributes = dict(self.attributes)

        # Map old blocks to new blocks
        block_map: dict[int, BasicBlock] = {}
        for old_block in self._blocks:
            new_block = BasicBlock(name=old_block.name, parent=new_func)
            block_map[old_block.id] = new_block
            new_func._blocks.append(new_block)

        # Map old values to new values
        val_map: dict[int, Value] = {}
        for i, old_arg in enumerate(self._arguments):
            val_map[old_arg.id] = new_func._arguments[i]

        # Clone instructions
        for old_block in self._blocks:
            new_block = block_map[old_block.id]
            for old_inst in old_block:
                new_inst = old_inst.clone()
                val_map[old_inst.id] = new_inst
                new_inst.parent = new_block
                new_block._instructions.append(new_inst)

        # Remap operands to new values
        for new_block_obj in new_func._blocks:
            for inst in new_block_obj:
                for i, op in enumerate(inst._operands):
                    if op.id in val_map:
                        mapped = val_map[op.id]
                        inst._operands[i] = mapped
                        mapped.users.add(inst)

                # Remap branch/switch targets
                if isinstance(inst, BranchInst):
                    inst._true_target = block_map.get(inst._true_target.id, inst._true_target)
                    if inst._false_target is not None:
                        inst._false_target = block_map.get(inst._false_target.id, inst._false_target)
                elif isinstance(inst, SwitchInst):
                    inst._default_target = block_map.get(inst._default_target.id, inst._default_target)
                    inst._cases = [
                        (c, block_map.get(b.id, b)) for c, b in inst._cases
                    ]
                elif isinstance(inst, PhiInst):
                    inst._incoming = [
                        (val_map.get(v.id, v), block_map.get(b.id, b))
                        for v, b in inst._incoming
                    ]

        # Rebuild CFG edges
        for new_block_obj in new_func._blocks:
            new_block_obj._update_successors()

        return new_func
