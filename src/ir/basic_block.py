"""
BasicBlock for the Cross-Language Equivalence Verifier IR.

Provides a container for a linear sequence of SSA instructions with
predecessor/successor tracking, phi node management, dominator tree
integration, and various iteration helpers.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Iterator, Optional, Sequence

from .instructions import (
    BranchInst,
    Instruction,
    PhiInst,
    ReturnInst,
    SwitchInst,
    Value,
    is_terminator,
    get_successors,
)

if TYPE_CHECKING:
    from .function import Function

_BLOCK_COUNTER = itertools.count()


def _fresh_block_id() -> int:
    return next(_BLOCK_COUNTER)


def reset_block_counter(start: int = 0) -> None:
    global _BLOCK_COUNTER
    _BLOCK_COUNTER = itertools.count(start)


class BasicBlock:
    """A basic block – a maximal straight-line sequence of instructions
    ending with exactly one terminator.

    Attributes:
        name: human-readable label for the block.
        parent: owning Function (set when the block is added to a function).
    """
    __slots__ = (
        "id", "name", "parent",
        "_instructions", "_predecessors", "_successors",
        "_idom", "_dominance_frontier", "_dom_children",
    )

    def __init__(self, name: str = "", parent: "Function | None" = None) -> None:
        self.id: int = _fresh_block_id()
        self.name: str = name or f"bb{self.id}"
        self.parent: Optional[Function] = parent

        self._instructions: list[Instruction] = []
        self._predecessors: list[BasicBlock] = []
        self._successors: list[BasicBlock] = []

        # Dominator tree fields – computed externally
        self._idom: Optional[BasicBlock] = None
        self._dominance_frontier: list[BasicBlock] = []
        self._dom_children: list[BasicBlock] = []

    # ------------------------------------------------------------------
    # Instruction container
    # ------------------------------------------------------------------

    @property
    def instructions(self) -> list[Instruction]:
        return list(self._instructions)

    def __len__(self) -> int:
        return len(self._instructions)

    def __iter__(self) -> Iterator[Instruction]:
        return iter(self._instructions)

    def __getitem__(self, index: int) -> Instruction:
        return self._instructions[index]

    def __contains__(self, inst: Instruction) -> bool:
        return inst in self._instructions

    @property
    def is_empty(self) -> bool:
        return len(self._instructions) == 0

    @property
    def first(self) -> Instruction | None:
        return self._instructions[0] if self._instructions else None

    @property
    def last(self) -> Instruction | None:
        return self._instructions[-1] if self._instructions else None

    @property
    def terminator(self) -> Instruction | None:
        """Return the terminator instruction, or None."""
        if self._instructions and is_terminator(self._instructions[-1]):
            return self._instructions[-1]
        return None

    @property
    def has_terminator(self) -> bool:
        return self.terminator is not None

    # ------------------------------------------------------------------
    # Insertion / removal
    # ------------------------------------------------------------------

    def append(self, inst: Instruction) -> None:
        """Append *inst* to the end of the block."""
        if inst.parent is not None:
            raise ValueError(f"Instruction {inst} already belongs to block {inst.parent.name}")
        inst.parent = self
        self._instructions.append(inst)
        # Update CFG edges if this is a terminator
        if is_terminator(inst):
            self._update_successors()

    def insert(self, index: int, inst: Instruction) -> None:
        """Insert *inst* at position *index*."""
        if inst.parent is not None:
            raise ValueError(f"Instruction {inst} already belongs to block {inst.parent.name}")
        inst.parent = self
        self._instructions.insert(index, inst)
        if is_terminator(inst):
            self._update_successors()

    def insert_before(self, before: Instruction, inst: Instruction) -> None:
        """Insert *inst* immediately before *before*."""
        idx = self._instructions.index(before)
        self.insert(idx, inst)

    def insert_after(self, after: Instruction, inst: Instruction) -> None:
        """Insert *inst* immediately after *after*."""
        idx = self._instructions.index(after) + 1
        self.insert(idx, inst)

    def insert_at_front(self, inst: Instruction) -> None:
        """Insert *inst* at the very beginning (before phi nodes too)."""
        self.insert(0, inst)

    def insert_before_terminator(self, inst: Instruction) -> None:
        """Insert *inst* just before the terminator (or at end if none)."""
        term = self.terminator
        if term is not None:
            self.insert_before(term, inst)
        else:
            self.append(inst)

    def remove(self, inst: Instruction) -> None:
        """Remove *inst* from this block."""
        self._instructions.remove(inst)
        inst.parent = None
        if is_terminator(inst):
            self._update_successors()

    def replace(self, old: Instruction, new: Instruction) -> None:
        """Replace *old* with *new* in-place."""
        idx = self._instructions.index(old)
        old.parent = None
        new.parent = self
        self._instructions[idx] = new
        old.replace_all_uses_with(new)
        if is_terminator(old) or is_terminator(new):
            self._update_successors()

    def clear(self) -> None:
        """Remove all instructions."""
        for inst in self._instructions:
            inst.parent = None
        self._instructions.clear()
        self._clear_successor_edges()

    # ------------------------------------------------------------------
    # Phi node helpers
    # ------------------------------------------------------------------

    @property
    def phi_nodes(self) -> list[PhiInst]:
        """Return all phi instructions at the start of this block."""
        result: list[PhiInst] = []
        for inst in self._instructions:
            if isinstance(inst, PhiInst):
                result.append(inst)
            else:
                break
        return result

    @property
    def num_phi_nodes(self) -> int:
        return len(self.phi_nodes)

    @property
    def non_phi_instructions(self) -> list[Instruction]:
        """Return all instructions after the phi nodes."""
        in_phi = True
        result: list[Instruction] = []
        for inst in self._instructions:
            if in_phi and isinstance(inst, PhiInst):
                continue
            in_phi = False
            result.append(inst)
        return result

    @property
    def first_non_phi(self) -> Instruction | None:
        for inst in self._instructions:
            if not isinstance(inst, PhiInst):
                return inst
        return None

    def add_phi(self, phi: PhiInst) -> None:
        """Add a phi node at the correct position (before non-phi instructions)."""
        insert_idx = 0
        for i, inst in enumerate(self._instructions):
            if not isinstance(inst, PhiInst):
                insert_idx = i
                break
            insert_idx = i + 1
        self.insert(insert_idx, phi)

    # ------------------------------------------------------------------
    # CFG edges
    # ------------------------------------------------------------------

    @property
    def predecessors(self) -> list["BasicBlock"]:
        return list(self._predecessors)

    @property
    def successors(self) -> list["BasicBlock"]:
        return list(self._successors)

    @property
    def num_predecessors(self) -> int:
        return len(self._predecessors)

    @property
    def num_successors(self) -> int:
        return len(self._successors)

    @property
    def is_entry(self) -> bool:
        """True if this block has no predecessors (potential entry block)."""
        return len(self._predecessors) == 0

    @property
    def is_exit(self) -> bool:
        """True if the terminator is a return instruction."""
        t = self.terminator
        return isinstance(t, ReturnInst)

    def add_predecessor(self, block: "BasicBlock") -> None:
        if block not in self._predecessors:
            self._predecessors.append(block)

    def remove_predecessor(self, block: "BasicBlock") -> None:
        if block in self._predecessors:
            self._predecessors.remove(block)
            # Remove phi incoming for this predecessor
            for phi in self.phi_nodes:
                phi.remove_incoming_block(block)

    def _update_successors(self) -> None:
        """Recompute successor edges from the terminator."""
        self._clear_successor_edges()
        term = self.terminator
        if term is not None:
            new_succs = get_successors(term)
            for succ in new_succs:
                if succ not in self._successors:
                    self._successors.append(succ)
                succ.add_predecessor(self)

    def _clear_successor_edges(self) -> None:
        for succ in self._successors:
            succ.remove_predecessor(self)
        self._successors.clear()

    # ------------------------------------------------------------------
    # Dominator tree integration
    # ------------------------------------------------------------------

    @property
    def idom(self) -> "BasicBlock | None":
        """Immediate dominator (set by dominator tree computation)."""
        return self._idom

    @idom.setter
    def idom(self, block: "BasicBlock | None") -> None:
        self._idom = block

    @property
    def dominance_frontier(self) -> list["BasicBlock"]:
        return self._dominance_frontier

    @dominance_frontier.setter
    def dominance_frontier(self, blocks: list["BasicBlock"]) -> None:
        self._dominance_frontier = blocks

    @property
    def dom_children(self) -> list["BasicBlock"]:
        return self._dom_children

    @dom_children.setter
    def dom_children(self, blocks: list["BasicBlock"]) -> None:
        self._dom_children = blocks

    def dominates(self, other: "BasicBlock") -> bool:
        """Return True if *self* dominates *other* (walks idom chain)."""
        cursor: BasicBlock | None = other
        while cursor is not None:
            if cursor is self:
                return True
            cursor = cursor._idom
        return False

    def strictly_dominates(self, other: "BasicBlock") -> bool:
        return self is not other and self.dominates(other)

    # ------------------------------------------------------------------
    # Iteration helpers
    # ------------------------------------------------------------------

    def iter_instructions(self) -> Iterator[Instruction]:
        """Iterate over instructions in program order."""
        return iter(self._instructions)

    def iter_instructions_reversed(self) -> Iterator[Instruction]:
        """Iterate over instructions in reverse program order."""
        return reversed(self._instructions)

    def iter_uses_of(self, value: Value) -> Iterator[Instruction]:
        """Iterate over instructions in this block that use *value*."""
        for inst in self._instructions:
            if value in inst.operands:
                yield inst

    def iter_definitions(self) -> Iterator[Instruction]:
        """Iterate over instructions that define a non-void value."""
        for inst in self._instructions:
            if not isinstance(inst.type, type) and not inst.type.is_void():
                yield inst

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Validate the block structure, returning error messages."""
        errors: list[str] = []

        if not self._instructions:
            errors.append(f"Block '{self.name}' is empty")
            return errors

        # Phi nodes must come first
        seen_non_phi = False
        for inst in self._instructions:
            if isinstance(inst, PhiInst):
                if seen_non_phi:
                    errors.append(f"Block '{self.name}': phi node after non-phi instruction")
            else:
                seen_non_phi = True

        # Must end with a terminator
        if not self.has_terminator:
            errors.append(f"Block '{self.name}' does not end with a terminator")

        # Only the last instruction may be a terminator
        for inst in self._instructions[:-1]:
            if is_terminator(inst):
                errors.append(f"Block '{self.name}': terminator in middle of block: {inst}")

        # Validate each instruction
        for inst in self._instructions:
            inst_errors = inst.validate()
            for e in inst_errors:
                errors.append(f"Block '{self.name}': {e}")

        return errors

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def split_at(self, inst: Instruction) -> "BasicBlock":
        """Split this block into two at *inst*.

        All instructions from *inst* onward are moved to a new block.
        An unconditional branch to the new block is added to *self*.

        Returns the new block.
        """
        idx = self._instructions.index(inst)
        new_block = BasicBlock(name=f"{self.name}.split", parent=self.parent)

        # Move instructions
        moving = self._instructions[idx:]
        self._instructions = self._instructions[:idx]
        for moved_inst in moving:
            moved_inst.parent = new_block
            new_block._instructions.append(moved_inst)

        # Update successor edges
        self._clear_successor_edges()
        new_block._update_successors()

        # Add branch from self to new_block
        from .instructions import BranchInst
        br = BranchInst(target=new_block)
        self.append(br)

        # Update phi nodes in successors of new_block
        for succ in new_block.successors:
            for phi in succ.phi_nodes:
                for i, (v, b) in enumerate(phi._incoming):
                    if b is self:
                        phi._incoming[i] = (v, new_block)

        # Add to parent function if possible
        if self.parent is not None:
            my_idx = self.parent._blocks.index(self)
            self.parent._blocks.insert(my_idx + 1, new_block)

        return new_block

    # ------------------------------------------------------------------
    # Printing
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"BasicBlock({self.name}, {len(self._instructions)} instructions)"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BasicBlock):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    def dump(self) -> str:
        """Return a multi-line string representation of the block."""
        lines = [f"{self.name}:"]
        preds = ", ".join(b.name for b in self._predecessors)
        if preds:
            lines.append(f"  ; predecessors: {preds}")
        for inst in self._instructions:
            lines.append(f"  {inst}")
        return "\n".join(lines)
