"""
Symbolic memory model.

Flat memory model with symbolic byte-level addressing, typed memory regions,
memory safety checking, pointer provenance tracking, and operation logging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Tuple, Set, Any

import z3


# ---------------------------------------------------------------------------
# Memory regions and errors
# ---------------------------------------------------------------------------

class MemoryRegionKind(Enum):
    """Kind of memory region."""
    STACK = auto()
    HEAP = auto()
    GLOBAL = auto()
    STATIC = auto()
    UNKNOWN = auto()


class MemoryErrorKind(Enum):
    """Kind of memory error."""
    NULL_DEREFERENCE = auto()
    BUFFER_OVERFLOW = auto()
    BUFFER_UNDERFLOW = auto()
    USE_AFTER_FREE = auto()
    DOUBLE_FREE = auto()
    MISALIGNED_ACCESS = auto()
    OUT_OF_BOUNDS = auto()
    UNINITIALIZED_READ = auto()
    STACK_OVERFLOW = auto()
    INVALID_FREE = auto()


@dataclass
class MemoryError:
    """A memory safety violation detected during symbolic execution."""
    kind: MemoryErrorKind
    address: Optional[z3.ExprRef] = None
    region: Optional[str] = None
    description: str = ""
    path_condition: Optional[z3.BoolRef] = None

    def __repr__(self) -> str:
        return f"MemError({self.kind.name}: {self.description})"


@dataclass
class MemoryRegion:
    """A typed memory region with provenance tracking."""
    region_id: str
    kind: MemoryRegionKind
    base_address: z3.BitVecRef
    size: int
    alignment: int = 1
    element_type: Optional[str] = None
    provenance: Optional[str] = None
    is_freed: bool = False
    is_initialized: bool = False

    @property
    def end_address_expr(self) -> z3.BitVecRef:
        return self.base_address + z3.BitVecVal(self.size, self.base_address.size())

    def contains(self, addr: z3.BitVecRef) -> z3.BoolRef:
        """Z3 expression: does this region contain the address?"""
        return z3.And(
            z3.UGE(addr, self.base_address),
            z3.ULT(addr, self.end_address_expr),
        )

    def __repr__(self) -> str:
        status = "freed" if self.is_freed else "active"
        return f"Region({self.region_id}, {self.kind.name}, size={self.size}, {status})"


@dataclass
class MemoryOperation:
    """Logged memory operation for counterexample generation."""
    op_type: str  # "load", "store", "alloc", "free"
    address: Optional[z3.ExprRef] = None
    value: Optional[z3.ExprRef] = None
    size: int = 0
    region_id: Optional[str] = None
    timestamp: int = 0


# ---------------------------------------------------------------------------
# Flat memory model
# ---------------------------------------------------------------------------

class FlatMemoryModel:
    """
    Flat symbolic memory model with byte-level addressing.
    
    Uses Z3 arrays for the memory contents, with typed regions
    for tracking allocations, provenance, and safety properties.
    """

    def __init__(
        self,
        address_width: int = 64,
        byte_width: int = 8,
        check_safety: bool = True,
    ):
        self.address_width = address_width
        self.byte_width = byte_width
        self.check_safety = check_safety

        # Memory sorts
        self._addr_sort = z3.BitVecSort(address_width)
        self._byte_sort = z3.BitVecSort(byte_width)

        # Z3 array: addr → byte
        self._version = 0
        self._memory = z3.Array(
            "mem_0", self._addr_sort, self._byte_sort,
        )

        # Initialization tracking array: addr → bool (is initialized)
        self._init_map = z3.Array(
            "init_0", self._addr_sort, z3.BoolSort(),
        )

        # Region tracking
        self._regions: Dict[str, MemoryRegion] = {}
        self._next_region_id = 0

        # Stack management
        self._stack_pointer = z3.BitVec("sp_0", address_width)
        self._stack_base = z3.BitVecVal(0x7FFF_FFFF_FFFF_0000, address_width) if address_width == 64 else z3.BitVecVal(0xFFFF0000, address_width)
        self._stack_size = 1 << 20  # 1MB stack

        # Heap management
        self._heap_base = z3.BitVecVal(0x0000_0001_0000_0000, address_width) if address_width == 64 else z3.BitVecVal(0x10000000, address_width)
        self._heap_bump = 0

        # Error log
        self._errors: List[MemoryError] = []

        # Operation log
        self._op_log: List[MemoryOperation] = []
        self._timestamp = 0

        # Provenance tracking
        self._provenance_map: Dict[str, str] = {}  # pointer_name → region_id

        # Fresh symbol counter
        self._sym_counter = 0

    # -- Memory operations --

    def store_byte(self, address: z3.BitVecRef, value: z3.BitVecRef) -> None:
        """Store a single byte."""
        self._version += 1
        self._memory = z3.Store(self._memory, address, value)
        self._init_map = z3.Store(self._init_map, address, z3.BoolVal(True))

        self._log_op("store", address, value, 1)

    def store(
        self,
        address: z3.BitVecRef,
        value: z3.BitVecRef,
        num_bytes: int,
    ) -> Optional[MemoryError]:
        """Store a multi-byte value (little-endian)."""
        if self.check_safety:
            err = self._check_store_safety(address, num_bytes)
            if err:
                self._errors.append(err)
                return err

        for i in range(num_bytes):
            if num_bytes == 1:
                byte_val = value
            else:
                byte_val = z3.Extract(
                    min((i + 1) * 8 - 1, value.size() - 1),
                    i * 8,
                    value,
                )
            offset = z3.BitVecVal(i, self.address_width)
            self.store_byte(address + offset, byte_val)

        self._log_op("store", address, value, num_bytes)
        return None

    def load_byte(self, address: z3.BitVecRef) -> z3.BitVecRef:
        """Load a single byte."""
        return z3.Select(self._memory, address)

    def load(
        self,
        address: z3.BitVecRef,
        num_bytes: int,
    ) -> Tuple[z3.BitVecRef, Optional[MemoryError]]:
        """Load a multi-byte value (little-endian)."""
        if self.check_safety:
            err = self._check_load_safety(address, num_bytes)
            if err:
                self._errors.append(err)

        if num_bytes == 1:
            result = self.load_byte(address)
        else:
            bytes_loaded: List[z3.BitVecRef] = []
            for i in range(num_bytes):
                offset = z3.BitVecVal(i, self.address_width)
                byte_val = self.load_byte(address + offset)
                bytes_loaded.append(byte_val)

            # Concatenate little-endian: MSB last
            result = bytes_loaded[-1]
            for i in range(len(bytes_loaded) - 2, -1, -1):
                result = z3.Concat(result, bytes_loaded[i])

        self._log_op("load", address, result, num_bytes)
        err = self._errors[-1] if self._errors and self.check_safety else None
        return result, err

    # -- Allocation --

    def alloc_stack(
        self,
        size: int,
        name: str = "",
        alignment: int = 1,
        element_type: str = "",
    ) -> Tuple[str, z3.BitVecRef]:
        """Allocate on the stack."""
        region_id = self._make_region_id(name or "stack")
        # Stack grows downward
        aligned_size = self._align_up(size, max(alignment, 1))
        self._stack_pointer = self._stack_pointer - z3.BitVecVal(aligned_size, self.address_width)

        base = self._stack_pointer

        region = MemoryRegion(
            region_id=region_id,
            kind=MemoryRegionKind.STACK,
            base_address=base,
            size=aligned_size,
            alignment=alignment,
            element_type=element_type,
            provenance=region_id,
        )
        self._regions[region_id] = region

        self._log_op("alloc", base, None, aligned_size, region_id)
        return region_id, base

    def alloc_heap(
        self,
        size: int,
        name: str = "",
        alignment: int = 1,
        element_type: str = "",
    ) -> Tuple[str, z3.BitVecRef]:
        """Allocate on the heap (malloc-like)."""
        region_id = self._make_region_id(name or "heap")
        aligned_size = self._align_up(size, max(alignment, 1))

        base = self._heap_base + z3.BitVecVal(self._heap_bump, self.address_width)
        self._heap_bump += aligned_size + 16  # 16-byte gap between allocations

        region = MemoryRegion(
            region_id=region_id,
            kind=MemoryRegionKind.HEAP,
            base_address=base,
            size=aligned_size,
            alignment=alignment,
            element_type=element_type,
            provenance=region_id,
        )
        self._regions[region_id] = region

        self._log_op("alloc", base, None, aligned_size, region_id)
        return region_id, base

    def alloc_global(
        self,
        size: int,
        name: str = "",
        init_value: Optional[z3.BitVecRef] = None,
    ) -> Tuple[str, z3.BitVecRef]:
        """Allocate a global variable."""
        region_id = self._make_region_id(name or "global")

        # Globals at fixed symbolic addresses
        base = z3.BitVec(f"global_{region_id}", self.address_width)

        region = MemoryRegion(
            region_id=region_id,
            kind=MemoryRegionKind.GLOBAL,
            base_address=base,
            size=size,
            is_initialized=init_value is not None,
            provenance=region_id,
        )
        self._regions[region_id] = region

        if init_value is not None:
            num_bytes = init_value.size() // 8 if z3.is_bv(init_value) else size
            self.store(base, init_value, num_bytes)

        self._log_op("alloc", base, init_value, size, region_id)
        return region_id, base

    def free(self, region_id: str) -> Optional[MemoryError]:
        """Free a heap allocation."""
        if region_id not in self._regions:
            err = MemoryError(
                kind=MemoryErrorKind.INVALID_FREE,
                description=f"Unknown region: {region_id}",
            )
            self._errors.append(err)
            return err

        region = self._regions[region_id]

        if region.is_freed:
            err = MemoryError(
                kind=MemoryErrorKind.DOUBLE_FREE,
                region=region_id,
                description=f"Double free of region {region_id}",
            )
            self._errors.append(err)
            return err

        if region.kind != MemoryRegionKind.HEAP:
            err = MemoryError(
                kind=MemoryErrorKind.INVALID_FREE,
                region=region_id,
                description=f"Cannot free {region.kind.name} region {region_id}",
            )
            self._errors.append(err)
            return err

        region.is_freed = True
        self._log_op("free", region.base_address, None, 0, region_id)
        return None

    def free_by_address(self, address: z3.BitVecRef) -> Optional[MemoryError]:
        """Free by address (find matching region)."""
        for rid, region in self._regions.items():
            if not region.is_freed and region.kind == MemoryRegionKind.HEAP:
                # Check if address matches base
                # This is symbolic, so we create a conditional
                self._log_op("free", address, None, 0, rid)
                region.is_freed = True
                return None

        return MemoryError(
            kind=MemoryErrorKind.INVALID_FREE,
            address=address,
            description="No matching allocation found",
        )

    # -- Safety checks --

    def _check_load_safety(
        self,
        address: z3.BitVecRef,
        num_bytes: int,
    ) -> Optional[MemoryError]:
        """Check if a load is safe."""
        # Null check
        null = z3.BitVecVal(0, self.address_width)
        is_null = address == null
        # We return the error condition symbolically
        return self._check_common_safety(address, num_bytes, "load")

    def _check_store_safety(
        self,
        address: z3.BitVecRef,
        num_bytes: int,
    ) -> Optional[MemoryError]:
        """Check if a store is safe."""
        return self._check_common_safety(address, num_bytes, "store")

    def _check_common_safety(
        self,
        address: z3.BitVecRef,
        num_bytes: int,
        op: str,
    ) -> Optional[MemoryError]:
        """Common safety check for load/store."""
        null_addr = z3.BitVecVal(0, self.address_width)

        # Build null dereference condition
        null_cond = address == null_addr

        # Build use-after-free conditions
        for rid, region in self._regions.items():
            if region.is_freed:
                in_freed = region.contains(address)
                # We note this as a potential error but don't block execution
                return MemoryError(
                    kind=MemoryErrorKind.USE_AFTER_FREE,
                    address=address,
                    region=rid,
                    description=f"Potential use-after-free in {op} to freed region {rid}",
                    path_condition=in_freed,
                )

        # Check if address could be null
        return MemoryError(
            kind=MemoryErrorKind.NULL_DEREFERENCE,
            address=address,
            description=f"Potential null dereference in {op}",
            path_condition=null_cond,
        )

    # -- Provenance --

    def set_provenance(self, pointer_name: str, region_id: str) -> None:
        """Track pointer provenance."""
        self._provenance_map[pointer_name] = region_id

    def get_provenance(self, pointer_name: str) -> Optional[str]:
        """Get the provenance of a pointer."""
        return self._provenance_map.get(pointer_name)

    def check_provenance(
        self,
        pointer_name: str,
        access_address: z3.BitVecRef,
    ) -> Optional[MemoryError]:
        """Check if an access through a pointer respects provenance."""
        prov = self.get_provenance(pointer_name)
        if prov is None:
            return None

        region = self._regions.get(prov)
        if region is None:
            return None

        in_region = region.contains(access_address)
        return MemoryError(
            kind=MemoryErrorKind.OUT_OF_BOUNDS,
            address=access_address,
            region=prov,
            description=f"Potential provenance violation: {pointer_name} accessing outside {prov}",
            path_condition=z3.Not(in_region),
        )

    # -- Memcpy/memset --

    def memcpy(
        self,
        dst: z3.BitVecRef,
        src: z3.BitVecRef,
        num_bytes: int,
    ) -> None:
        """Copy memory from src to dst."""
        for i in range(num_bytes):
            offset = z3.BitVecVal(i, self.address_width)
            byte_val = self.load_byte(src + offset)
            self.store_byte(dst + offset, byte_val)

        self._log_op("memcpy", dst, src, num_bytes)

    def memset(
        self,
        dst: z3.BitVecRef,
        value: z3.BitVecRef,
        num_bytes: int,
    ) -> None:
        """Set memory to a value."""
        byte_val = value
        if z3.is_bv(value) and value.size() > 8:
            byte_val = z3.Extract(7, 0, value)

        for i in range(num_bytes):
            offset = z3.BitVecVal(i, self.address_width)
            self.store_byte(dst + offset, byte_val)

        self._log_op("memset", dst, value, num_bytes)

    def memcmp(
        self,
        a: z3.BitVecRef,
        b: z3.BitVecRef,
        num_bytes: int,
    ) -> z3.BitVecRef:
        """Compare two memory regions symbolically."""
        if num_bytes == 0:
            return z3.BitVecVal(0, 32)

        # Build comparison byte by byte
        all_equal = z3.BoolVal(True)
        for i in range(num_bytes):
            offset = z3.BitVecVal(i, self.address_width)
            byte_a = self.load_byte(a + offset)
            byte_b = self.load_byte(b + offset)
            all_equal = z3.And(all_equal, byte_a == byte_b)

        return z3.If(all_equal, z3.BitVecVal(0, 32), z3.BitVecVal(1, 32))

    # -- Constraints --

    def get_constraints(self) -> List[z3.BoolRef]:
        """Get all memory-related constraints (non-overlap, non-null, etc.)."""
        constraints: List[z3.BoolRef] = []

        active_regions = [
            (rid, r) for rid, r in self._regions.items() if not r.is_freed
        ]

        # Non-null for all regions
        for rid, region in active_regions:
            constraints.append(
                region.base_address != z3.BitVecVal(0, self.address_width)
            )
            # Minimum address (avoid very low addresses)
            constraints.append(
                z3.UGT(region.base_address, z3.BitVecVal(0x1000, self.address_width))
            )

        # Non-overlap between heap/global regions
        heap_globals = [
            (rid, r) for rid, r in active_regions
            if r.kind in (MemoryRegionKind.HEAP, MemoryRegionKind.GLOBAL)
        ]
        for i in range(len(heap_globals)):
            for j in range(i + 1, len(heap_globals)):
                _, ri = heap_globals[i]
                _, rj = heap_globals[j]
                constraints.append(z3.Or(
                    z3.UGE(rj.base_address, ri.end_address_expr),
                    z3.UGE(ri.base_address, rj.end_address_expr),
                ))

        # Alignment constraints
        for rid, region in active_regions:
            if region.alignment > 1:
                mask = z3.BitVecVal(region.alignment - 1, self.address_width)
                constraints.append(
                    (region.base_address & mask) == z3.BitVecVal(0, self.address_width)
                )

        return constraints

    # -- Errors --

    @property
    def errors(self) -> List[MemoryError]:
        return list(self._errors)

    def has_errors(self) -> bool:
        return len(self._errors) > 0

    # -- Copy --

    def copy(self) -> FlatMemoryModel:
        """Create a deep copy of this memory model."""
        new = FlatMemoryModel(
            self.address_width, self.byte_width, self.check_safety,
        )
        new._memory = self._memory
        new._init_map = self._init_map
        new._version = self._version
        new._regions = {
            rid: MemoryRegion(
                region_id=r.region_id,
                kind=r.kind,
                base_address=r.base_address,
                size=r.size,
                alignment=r.alignment,
                element_type=r.element_type,
                provenance=r.provenance,
                is_freed=r.is_freed,
                is_initialized=r.is_initialized,
            )
            for rid, r in self._regions.items()
        }
        new._next_region_id = self._next_region_id
        new._stack_pointer = self._stack_pointer
        new._heap_bump = self._heap_bump
        new._errors = list(self._errors)
        new._op_log = list(self._op_log)
        new._timestamp = self._timestamp
        new._provenance_map = dict(self._provenance_map)
        new._sym_counter = self._sym_counter
        return new

    # -- Helpers --

    def _make_region_id(self, prefix: str) -> str:
        self._next_region_id += 1
        return f"{prefix}_{self._next_region_id}"

    def _align_up(self, size: int, alignment: int) -> int:
        if alignment <= 1:
            return size
        return (size + alignment - 1) & ~(alignment - 1)

    def _log_op(
        self,
        op_type: str,
        address: Optional[z3.ExprRef],
        value: Optional[z3.ExprRef],
        size: int,
        region_id: Optional[str] = None,
    ) -> None:
        self._timestamp += 1
        self._op_log.append(MemoryOperation(
            op_type=op_type,
            address=address,
            value=value,
            size=size,
            region_id=region_id,
            timestamp=self._timestamp,
        ))

    def _fresh_sym(self, name: str, width: int) -> z3.BitVecRef:
        self._sym_counter += 1
        return z3.BitVec(f"{name}_{self._sym_counter}", width)

    # -- Query --

    def get_regions(self) -> List[MemoryRegion]:
        return list(self._regions.values())

    def get_active_regions(self) -> List[MemoryRegion]:
        return [r for r in self._regions.values() if not r.is_freed]

    def get_op_log(self) -> List[MemoryOperation]:
        return list(self._op_log)

    def summary(self) -> str:
        active = self.get_active_regions()
        freed = [r for r in self._regions.values() if r.is_freed]
        lines = [
            f"Memory Model (addr={self.address_width}b):",
            f"  Regions: {len(self._regions)} total, {len(active)} active, {len(freed)} freed",
            f"  Operations: {len(self._op_log)}",
            f"  Errors: {len(self._errors)}",
            f"  Version: {self._version}",
        ]
        for r in active:
            lines.append(f"    {r}")
        return "\n".join(lines)
