"""
Product program construction.

Given two aligned IR functions, produces a single product program with shared
symbolic inputs, parallel execution paths, and coercion assertions at
divergence points.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Tuple, Set, Any, Iterator

from ..ir.types import (
    IRType, IntType, FloatType, PointerType, VoidType,
    StructType, ArrayType, FunctionType, Signedness,
    OverflowBehavior, Language,
)
from ..ir.instructions import (
    Instruction, BinaryOp, UnaryOp, CompareOp, CastInst,
    LoadInst, StoreInst, CallInst, ReturnInst, BranchInst,
    PhiInst, SelectInst, AllocaInst, GetElementPtrInst,
    Value, Constant, Argument, BinOpKind,
)
from ..ir.basic_block import BasicBlock
from ..ir.function import Function
from ..semantics.divergence_table import DivergenceTable, DivergenceClass
from ..semantics.semantic_config import SemanticConfig, OverflowMode
from .alignment import (
    AlignmentResult, BlockAlignment, InstructionAlignment,
    AlignmentKind, FunctionAligner,
)
from .coercion import (
    CoercionPoint, CoercionKind, CoercionGenerator,
    CoercionAssertion, AssertionStrength,
)


# ---------------------------------------------------------------------------
# Product instruction: wraps a pair of aligned instructions
# ---------------------------------------------------------------------------

class ProductSide(Enum):
    LEFT = auto()
    RIGHT = auto()
    BOTH = auto()


@dataclass
class SharedInput:
    """A symbolic input shared between both sides of the product program."""
    name: str
    ir_type: IRType
    left_arg_index: int
    right_arg_index: int

    def __repr__(self) -> str:
        return f"SharedInput({self.name}: {self.ir_type})"


@dataclass
class ProductInstruction:
    """An instruction in the product program, representing one or both sides."""
    left_inst: Optional[Instruction]
    right_inst: Optional[Instruction]
    side: ProductSide
    coercion_points: List[CoercionPoint] = field(default_factory=list)
    left_result_name: str = ""
    right_result_name: str = ""
    notes: List[str] = field(default_factory=list)

    @property
    def has_coercions(self) -> bool:
        return len(self.coercion_points) > 0

    @property
    def is_both(self) -> bool:
        return self.side == ProductSide.BOTH

    @property
    def all_assertions(self) -> List[CoercionAssertion]:
        result: List[CoercionAssertion] = []
        for cp in self.coercion_points:
            result.extend(cp.assertions)
        return result

    def summary(self) -> str:
        l = self.left_inst.name if self.left_inst else "---"
        r = self.right_inst.name if self.right_inst else "---"
        coercion_str = f" [{len(self.coercion_points)} coercions]" if self.has_coercions else ""
        return f"P({l} | {r}){coercion_str}"


# ---------------------------------------------------------------------------
# Product block
# ---------------------------------------------------------------------------

@dataclass
class ProductBlock:
    """A block in the product program, merging two aligned blocks."""
    name: str
    left_block: Optional[BasicBlock]
    right_block: Optional[BasicBlock]
    instructions: List[ProductInstruction] = field(default_factory=list)
    predecessors: List[str] = field(default_factory=list)
    successors: List[str] = field(default_factory=list)
    is_entry: bool = False
    is_exit: bool = False

    @property
    def coercion_points(self) -> List[CoercionPoint]:
        result: List[CoercionPoint] = []
        for pi in self.instructions:
            result.extend(pi.coercion_points)
        return result

    @property
    def num_instructions(self) -> int:
        return len(self.instructions)

    @property
    def has_coercions(self) -> bool:
        return any(pi.has_coercions for pi in self.instructions)

    def iter_left_instructions(self) -> Iterator[Instruction]:
        for pi in self.instructions:
            if pi.left_inst is not None:
                yield pi.left_inst

    def iter_right_instructions(self) -> Iterator[Instruction]:
        for pi in self.instructions:
            if pi.right_inst is not None:
                yield pi.right_inst


# ---------------------------------------------------------------------------
# Product program
# ---------------------------------------------------------------------------

@dataclass
class ProductProgram:
    """
    A product program combining two aligned functions.
    
    Contains shared symbolic inputs, parallel execution blocks,
    and coercion assertions at divergence points.
    """
    name: str
    left_function: Function
    right_function: Function
    shared_inputs: List[SharedInput] = field(default_factory=list)
    blocks: List[ProductBlock] = field(default_factory=list)
    coercion_points: List[CoercionPoint] = field(default_factory=list)
    alignment: Optional[AlignmentResult] = None
    c_config: Optional[SemanticConfig] = None
    rust_config: Optional[SemanticConfig] = None
    _block_map: Dict[str, ProductBlock] = field(default_factory=dict)

    def __post_init__(self):
        self._block_map = {b.name: b for b in self.blocks}

    @property
    def entry_block(self) -> Optional[ProductBlock]:
        for b in self.blocks:
            if b.is_entry:
                return b
        return self.blocks[0] if self.blocks else None

    @property
    def exit_blocks(self) -> List[ProductBlock]:
        return [b for b in self.blocks if b.is_exit]

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    @property
    def num_instructions(self) -> int:
        return sum(b.num_instructions for b in self.blocks)

    @property
    def num_coercion_points(self) -> int:
        return len(self.coercion_points)

    @property
    def num_assertions(self) -> int:
        return sum(cp.num_assertions for cp in self.coercion_points)

    def get_block(self, name: str) -> Optional[ProductBlock]:
        return self._block_map.get(name)

    def add_block(self, block: ProductBlock) -> None:
        self.blocks.append(block)
        self._block_map[block.name] = block

    def iter_instructions(self) -> Iterator[ProductInstruction]:
        for block in self.blocks:
            yield from block.instructions

    def iter_coercion_points(self) -> Iterator[CoercionPoint]:
        yield from self.coercion_points

    def all_assertions(self) -> List[CoercionAssertion]:
        result: List[CoercionAssertion] = []
        for cp in self.coercion_points:
            result.extend(cp.assertions)
        return result

    def hard_assertions(self) -> List[CoercionAssertion]:
        return [a for a in self.all_assertions() if a.strength == AssertionStrength.HARD]

    def assumptions(self) -> List[CoercionAssertion]:
        return [a for a in self.all_assertions() if a.strength == AssertionStrength.ASSUME]

    def summary(self) -> str:
        lines = [
            f"ProductProgram: {self.name}",
            f"  Left:  {self.left_function.name} ({len(list(self.left_function.blocks))} blocks)",
            f"  Right: {self.right_function.name} ({len(list(self.right_function.blocks))} blocks)",
            f"  Shared inputs: {len(self.shared_inputs)}",
            f"  Product blocks: {self.num_blocks}",
            f"  Product instructions: {self.num_instructions}",
            f"  Coercion points: {self.num_coercion_points}",
            f"  Total assertions: {self.num_assertions}",
            f"    Hard: {len(self.hard_assertions())}",
            f"    Assumptions: {len(self.assumptions())}",
        ]
        return "\n".join(lines)

    def visualize(self, max_width: int = 100) -> str:
        lines: List[str] = []
        lines.append("=" * max_width)
        lines.append(f"PRODUCT PROGRAM: {self.name}")
        lines.append("=" * max_width)

        lines.append("\nShared Inputs:")
        for si in self.shared_inputs:
            lines.append(f"  {si.name}: {si.ir_type}")

        lines.append(f"\nBlocks ({self.num_blocks}):")
        for block in self.blocks:
            flags = []
            if block.is_entry:
                flags.append("ENTRY")
            if block.is_exit:
                flags.append("EXIT")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            left_name = block.left_block.name if block.left_block else "---"
            right_name = block.right_block.name if block.right_block else "---"
            lines.append(f"\n  {block.name}{flag_str}  ({left_name} | {right_name})")
            lines.append(f"  preds: {block.predecessors}  succs: {block.successors}")

            for pi in block.instructions:
                coercion_mark = " ⚠" if pi.has_coercions else ""
                lines.append(f"    {pi.summary()}{coercion_mark}")
                for cp in pi.coercion_points:
                    lines.append(f"      COERCION: {cp.kind.name} - {cp.divergence_class.name}")
                    for a in cp.assertions:
                        lines.append(f"        [{a.strength.name}] {a.description}")

        lines.append("\n" + "=" * max_width)
        return "\n".join(lines)

    def to_smt_lib(self) -> str:
        """Generate SMT-LIB2 script for the product program assertions."""
        lines: List[str] = []
        lines.append("; Product program verification conditions")
        lines.append(f"; Left:  {self.left_function.name}")
        lines.append(f"; Right: {self.right_function.name}")
        lines.append("(set-logic QF_BV)")
        lines.append("")

        # Declare shared inputs
        for si in self.shared_inputs:
            sort = _ir_type_to_smt_sort(si.ir_type)
            lines.append(f"(declare-const {si.name} {sort})")

        lines.append("")

        # Collect all variable declarations from assertions
        declared: Set[str] = {si.name for si in self.shared_inputs}
        for cp in self.coercion_points:
            for a in cp.assertions:
                for v in a.variables:
                    if v not in declared:
                        lines.append(f"(declare-const {v} (_ BitVec {cp.bit_width}))")
                        declared.add(v)

        lines.append("")

        # Add assumptions
        assumptions = self.assumptions()
        if assumptions:
            lines.append("; Assumptions (preconditions)")
            for a in assumptions:
                lines.append(f"(assert {a.smt_expression})")
                lines.append(f"  ; {a.description}")
            lines.append("")

        # Add hard assertions (negated for counterexample search)
        hard = self.hard_assertions()
        if hard:
            lines.append("; Verification conditions (negated for counterexample search)")
            if len(hard) == 1:
                lines.append(f"(assert (not {hard[0].smt_expression}))")
            else:
                conj = "(and " + " ".join(a.smt_expression for a in hard) + ")"
                lines.append(f"(assert (not {conj}))")
            lines.append("")

        lines.append("(check-sat)")
        lines.append("(get-model)")
        return "\n".join(lines)


def _ir_type_to_smt_sort(ty: IRType) -> str:
    """Convert IR type to SMT-LIB2 sort."""
    if isinstance(ty, IntType):
        return f"(_ BitVec {ty.width})"
    if isinstance(ty, FloatType):
        if ty.kind == FloatKind.F32:
            return "(_ FloatingPoint 8 24)"
        return "(_ FloatingPoint 11 53)"
    if isinstance(ty, PointerType):
        return "(_ BitVec 64)"
    if isinstance(ty, VoidType):
        return "Bool"
    return "(_ BitVec 32)"


# Need FloatKind import
from ..ir.types import FloatKind


# ---------------------------------------------------------------------------
# Product program builder
# ---------------------------------------------------------------------------

class ProductBuilder:
    """
    Builds a product program from two IR functions.
    
    Steps:
    1. Align the two functions (block and instruction level)
    2. Create shared symbolic inputs
    3. Build product blocks with parallel execution
    4. Insert coercion assertions at divergence points
    """

    def __init__(
        self,
        divergence_table: Optional[DivergenceTable] = None,
        c_config: Optional[SemanticConfig] = None,
        rust_config: Optional[SemanticConfig] = None,
        aligner: Optional[FunctionAligner] = None,
    ):
        self.table = divergence_table or DivergenceTable()
        self.c_config = c_config or SemanticConfig.c11()
        self.rust_config = rust_config or SemanticConfig.rust_release()
        self.aligner = aligner or FunctionAligner()
        self.coercion_gen = CoercionGenerator(self.table, self.c_config, self.rust_config)
        self._name_counter = 0

    def _fresh_name(self, prefix: str = "pb") -> str:
        self._name_counter += 1
        return f"{prefix}_{self._name_counter}"

    def build(self, left: Function, right: Function) -> ProductProgram:
        """Build a product program from two functions."""
        # Step 1: Align
        alignment = self.aligner.align(left, right)

        # Step 2: Create shared inputs
        shared_inputs = self._create_shared_inputs(left, right)

        # Step 3: Build product program shell
        product = ProductProgram(
            name=f"product_{left.name}_x_{right.name}",
            left_function=left,
            right_function=right,
            shared_inputs=shared_inputs,
            alignment=alignment,
            c_config=self.c_config,
            rust_config=self.rust_config,
        )

        # Step 4: Build product blocks
        self._build_product_blocks(product, alignment)

        # Step 5: Generate coercion points
        coercion_points = self.coercion_gen.generate_for_alignment(alignment)
        product.coercion_points = coercion_points

        # Step 6: Distribute coercion points to product instructions
        self._distribute_coercions(product, coercion_points)

        # Step 7: Wire up block predecessors/successors
        self._wire_blocks(product, alignment)

        return product

    def _create_shared_inputs(
        self, left: Function, right: Function,
    ) -> List[SharedInput]:
        """Create shared symbolic inputs for the product program."""
        shared: List[SharedInput] = []

        left_args = list(left.arguments)
        right_args = list(right.arguments)

        min_args = min(len(left_args), len(right_args))

        for i in range(min_args):
            la = left_args[i]
            ra = right_args[i]

            # Choose the more general type
            shared_type = self._unify_types(la.type, ra.type)
            name = la.name or ra.name or f"arg_{i}"

            shared.append(SharedInput(
                name=f"input_{name}",
                ir_type=shared_type,
                left_arg_index=i,
                right_arg_index=i,
            ))

        # Extra left-only arguments
        for i in range(min_args, len(left_args)):
            la = left_args[i]
            shared.append(SharedInput(
                name=f"input_left_{la.name or i}",
                ir_type=la.type,
                left_arg_index=i,
                right_arg_index=-1,
            ))

        # Extra right-only arguments
        for i in range(min_args, len(right_args)):
            ra = right_args[i]
            shared.append(SharedInput(
                name=f"input_right_{ra.name or i}",
                ir_type=ra.type,
                left_arg_index=-1,
                right_arg_index=i,
            ))

        return shared

    def _unify_types(self, lt: IRType, rt: IRType) -> IRType:
        """Unify two types, choosing the more general one."""
        if lt == rt:
            return lt

        if isinstance(lt, IntType) and isinstance(rt, IntType):
            width = max(lt.width, rt.width)
            signed = lt.is_signed or rt.is_signed
            return IntType(width, Signedness.SIGNED if signed else Signedness.UNSIGNED)

        if isinstance(lt, FloatType) and isinstance(rt, FloatType):
            if lt.kind == FloatKind.F64 or rt.kind == FloatKind.F64:
                return FloatType(FloatKind.F64)
            return FloatType(FloatKind.F32)

        if isinstance(lt, PointerType) and isinstance(rt, PointerType):
            return lt  # Use left's pointer type

        # Fallback: use left type
        return lt

    def _build_product_blocks(
        self, product: ProductProgram, alignment: AlignmentResult,
    ) -> None:
        """Build product blocks from aligned blocks."""
        for i, ba in enumerate(alignment.block_alignments):
            block_name = self._fresh_name("pblock")

            pblock = ProductBlock(
                name=block_name,
                left_block=ba.left,
                right_block=ba.right,
                is_entry=(i == 0),
            )

            if ba.is_matched or ba.kind == AlignmentKind.REORDERED:
                self._build_matched_block(pblock, ba)
            elif ba.kind == AlignmentKind.LEFT_ONLY:
                self._build_left_only_block(pblock, ba)
            elif ba.kind == AlignmentKind.RIGHT_ONLY:
                self._build_right_only_block(pblock, ba)

            # Check if this is an exit block
            if ba.left and ba.right:
                left_is_exit = any(isinstance(i, ReturnInst) for i in ba.left.instructions)
                right_is_exit = any(isinstance(i, ReturnInst) for i in ba.right.instructions)
                pblock.is_exit = left_is_exit or right_is_exit
            elif ba.left:
                pblock.is_exit = any(isinstance(i, ReturnInst) for i in ba.left.instructions)
            elif ba.right:
                pblock.is_exit = any(isinstance(i, ReturnInst) for i in ba.right.instructions)

            product.add_block(pblock)

    def _build_matched_block(
        self, pblock: ProductBlock, ba: BlockAlignment,
    ) -> None:
        """Build product instructions for a matched block pair."""
        for ia in ba.instruction_alignments:
            if ia.is_matched:
                left_name = f"c_{ia.left.name}" if ia.left and ia.left.name else ""
                right_name = f"rust_{ia.right.name}" if ia.right and ia.right.name else ""

                pi = ProductInstruction(
                    left_inst=ia.left,
                    right_inst=ia.right,
                    side=ProductSide.BOTH,
                    left_result_name=left_name,
                    right_result_name=right_name,
                )
                pblock.instructions.append(pi)
            elif ia.is_left_only:
                pi = ProductInstruction(
                    left_inst=ia.left,
                    right_inst=None,
                    side=ProductSide.LEFT,
                    left_result_name=f"c_{ia.left.name}" if ia.left and ia.left.name else "",
                    notes=["left-only instruction"],
                )
                pblock.instructions.append(pi)
            elif ia.is_right_only:
                pi = ProductInstruction(
                    left_inst=None,
                    right_inst=ia.right,
                    side=ProductSide.RIGHT,
                    right_result_name=f"rust_{ia.right.name}" if ia.right and ia.right.name else "",
                    notes=["right-only instruction"],
                )
                pblock.instructions.append(pi)

    def _build_left_only_block(
        self, pblock: ProductBlock, ba: BlockAlignment,
    ) -> None:
        """Build product instructions for a left-only block."""
        if ba.left is None:
            return
        for inst in ba.left.instructions:
            pi = ProductInstruction(
                left_inst=inst,
                right_inst=None,
                side=ProductSide.LEFT,
                left_result_name=f"c_{inst.name}" if inst.name else "",
                notes=["left-only block"],
            )
            pblock.instructions.append(pi)

    def _build_right_only_block(
        self, pblock: ProductBlock, ba: BlockAlignment,
    ) -> None:
        """Build product instructions for a right-only block."""
        if ba.right is None:
            return
        for inst in ba.right.instructions:
            pi = ProductInstruction(
                left_inst=None,
                right_inst=inst,
                side=ProductSide.RIGHT,
                right_result_name=f"rust_{inst.name}" if inst.name else "",
                notes=["right-only block"],
            )
            pblock.instructions.append(pi)

    def _distribute_coercions(
        self, product: ProductProgram, coercion_points: List[CoercionPoint],
    ) -> None:
        """Distribute coercion points to their corresponding product instructions."""
        for cp in coercion_points:
            if cp.left_instruction is None and cp.right_instruction is None:
                # Return coercion: attach to exit blocks
                for eb in product.exit_blocks:
                    if eb.instructions:
                        eb.instructions[-1].coercion_points.append(cp)
                continue

            # Find the product instruction that corresponds to this coercion point
            for block in product.blocks:
                for pi in block.instructions:
                    if (pi.left_inst is cp.left_instruction and
                            pi.right_inst is cp.right_instruction):
                        pi.coercion_points.append(cp)
                        break

    def _wire_blocks(
        self, product: ProductProgram, alignment: AlignmentResult,
    ) -> None:
        """Wire product blocks with predecessor/successor relationships."""
        # Build name→index mapping for left and right blocks
        left_block_to_product: Dict[int, str] = {}
        right_block_to_product: Dict[int, str] = {}

        for pblock in product.blocks:
            if pblock.left_block is not None:
                left_block_to_product[id(pblock.left_block)] = pblock.name
            if pblock.right_block is not None:
                right_block_to_product[id(pblock.right_block)] = pblock.name

        for pblock in product.blocks:
            successors: Set[str] = set()

            # Get successors from left block
            if pblock.left_block is not None:
                for succ in pblock.left_block._successors:
                    succ_name = left_block_to_product.get(id(succ))
                    if succ_name:
                        successors.add(succ_name)

            # Get successors from right block
            if pblock.right_block is not None:
                for succ in pblock.right_block._successors:
                    succ_name = right_block_to_product.get(id(succ))
                    if succ_name:
                        successors.add(succ_name)

            pblock.successors = sorted(successors)

            # Set predecessor relationships
            for succ_name in pblock.successors:
                succ_block = product.get_block(succ_name)
                if succ_block and pblock.name not in succ_block.predecessors:
                    succ_block.predecessors.append(pblock.name)

    def build_with_overflow_encoding(
        self,
        left: Function,
        right: Function,
        c_overflow: OverflowMode = OverflowMode.UB,
        rust_overflow: OverflowMode = OverflowMode.Wrap,
    ) -> ProductProgram:
        """Build product program with explicit overflow mode encoding."""
        # Create configs with the specified overflow modes
        c_config = SemanticConfig.c11()
        c_config.signed_overflow = c_overflow
        rust_config = SemanticConfig.rust_release()
        rust_config.signed_overflow = rust_overflow

        old_c = self.c_config
        old_r = self.rust_config
        self.c_config = c_config
        self.rust_config = rust_config
        self.coercion_gen = CoercionGenerator(self.table, c_config, rust_config)

        product = self.build(left, right)

        self.c_config = old_c
        self.rust_config = old_r
        self.coercion_gen = CoercionGenerator(self.table, old_c, old_r)

        return product

    def build_with_error_handling(
        self,
        left: Function,
        right: Function,
    ) -> ProductProgram:
        """Build product program with enhanced error handling divergence detection."""
        product = self.build(left, right)

        # Add error handling coercions for call instructions
        for block in product.blocks:
            for pi in block.instructions:
                if pi.left_inst and pi.right_inst:
                    if isinstance(pi.left_inst, CallInst) and isinstance(pi.right_inst, CallInst):
                        # C error: return code, Rust error: Result/panic
                        err_cp = self._create_error_handling_coercion(
                            pi.left_inst, pi.right_inst
                        )
                        if err_cp:
                            pi.coercion_points.append(err_cp)
                            product.coercion_points.append(err_cp)

        return product

    def _create_error_handling_coercion(
        self,
        c_call: CallInst,
        rust_call: CallInst,
    ) -> Optional[CoercionPoint]:
        """Create a coercion point for error handling differences."""
        entry = self.table.get(DivergenceClass.ErrorHandling)
        if entry is None:
            return None

        prefix = f"err_{c_call.callee_name}"
        c_ret = f"{prefix}_c_ret"
        rust_ret = f"{prefix}_rust_ret"
        rust_panicked = f"{prefix}_rust_panicked"

        assertions = [
            CoercionAssertion(
                smt_expression=f"(=> (= {rust_panicked} (_ bv1 1)) (bvslt {c_ret} (_ bv0 32)))",
                description=f"Error mapping: if Rust panics, C should return error code",
                strength=AssertionStrength.SOFT,
                variables=[c_ret, rust_ret, rust_panicked],
            ),
            CoercionAssertion(
                smt_expression=f"(=> (bvsge {c_ret} (_ bv0 32)) (= {c_ret} {rust_ret}))",
                description=f"Success case: return values should match",
                strength=AssertionStrength.HARD,
                variables=[c_ret, rust_ret],
            ),
        ]

        return CoercionPoint(
            kind=CoercionKind.ERROR_HANDLING,
            left_instruction=c_call,
            right_instruction=rust_call,
            divergence_class=DivergenceClass.ErrorHandling,
            c_semantics=entry.c_semantics,
            rust_semantics=entry.rust_semantics,
            assertions=assertions,
            operation=f"CALL({c_call.callee_name})",
            bit_width=32,
        )
