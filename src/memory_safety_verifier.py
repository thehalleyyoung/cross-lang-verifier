"""
Memory safety verifier.
Verifies spatial safety, temporal safety, stack safety, heap analysis,
aliasing analysis, RAII verification, and buffer overflow gadget detection.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from enum import Enum, auto
import copy


# ---------------------------------------------------------------------------
# Safety violation types
# ---------------------------------------------------------------------------

class SafetyViolationType(Enum):
    SPATIAL_OUT_OF_BOUNDS = "spatial-out-of-bounds"
    SPATIAL_NEGATIVE_INDEX = "spatial-negative-index"
    TEMPORAL_USE_AFTER_FREE = "temporal-use-after-free"
    TEMPORAL_DOUBLE_FREE = "temporal-double-free"
    TEMPORAL_DANGLING_PTR = "temporal-dangling-pointer"
    STACK_BUFFER_OVERFLOW = "stack-buffer-overflow"
    STACK_RETURN_OVERWRITE = "stack-return-address-overwrite"
    STACK_UNDERFLOW = "stack-underflow"
    HEAP_LEAK = "heap-memory-leak"
    HEAP_OVERFLOW = "heap-buffer-overflow"
    HEAP_UNDERFLOW = "heap-buffer-underflow"
    ALIAS_CONFLICT = "aliasing-conflict-write"
    RAII_RESOURCE_LEAK = "raii-resource-leak"
    RAII_DOUBLE_DROP = "raii-double-drop"
    RAII_USE_AFTER_DROP = "raii-use-after-drop"
    EXPLOIT_STACK_SMASH = "exploitable-stack-smash"
    EXPLOIT_HEAP_OVERFLOW = "exploitable-heap-overflow"
    EXPLOIT_FORMAT_STRING = "exploitable-format-string"
    EXPLOIT_USE_AFTER_FREE = "exploitable-use-after-free"
    UNINIT_READ = "uninitialized-read"
    NULL_DEREF = "null-pointer-dereference"


class ExploitDifficulty(Enum):
    TRIVIAL = "trivial"
    EASY = "easy"
    MODERATE = "moderate"
    HARD = "hard"
    THEORETICAL = "theoretical"


@dataclass
class SafetyViolation:
    type: SafetyViolationType
    location: str
    description: str
    severity: str = "error"
    variable: str = ""
    suggestion: str = ""
    exploitable: bool = False
    exploit_difficulty: Optional[ExploitDifficulty] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "location": self.location,
            "description": self.description,
            "severity": self.severity,
            "variable": self.variable,
            "suggestion": self.suggestion,
            "exploitable": self.exploitable,
            "exploit_difficulty": self.exploit_difficulty.value if self.exploit_difficulty else None,
        }


@dataclass
class MemoryLeak:
    allocation_site: str
    variable: str
    size: Optional[int] = None
    reachable: bool = False
    leak_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allocation_site": self.allocation_site,
            "variable": self.variable,
            "size": self.size,
            "reachable": self.reachable,
            "leak_path": self.leak_path,
        }


@dataclass
class ExploitableBug:
    violation: SafetyViolation
    exploit_type: str
    difficulty: ExploitDifficulty
    description: str
    cwe_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "violation": self.violation.to_dict(),
            "exploit_type": self.exploit_type,
            "difficulty": self.difficulty.value,
            "description": self.description,
            "cwe_id": self.cwe_id,
        }


@dataclass
class Remediation:
    violation_type: SafetyViolationType
    suggestion: str
    code_example: str = ""
    priority: str = "high"


@dataclass
class SafetyReport:
    safe: bool = True
    violations: List[SafetyViolation] = field(default_factory=list)
    memory_leaks: List[MemoryLeak] = field(default_factory=list)
    exploitable_bugs: List[ExploitableBug] = field(default_factory=list)
    remediation: List[Remediation] = field(default_factory=list)
    stats: Dict[str, int] = field(default_factory=dict)
    language: str = ""

    def add_violation(self, v: SafetyViolation) -> None:
        self.violations.append(v)
        if v.severity == "error":
            self.safe = False

    def add_leak(self, leak: MemoryLeak) -> None:
        self.memory_leaks.append(leak)
        self.safe = False

    def add_exploit(self, exploit: ExploitableBug) -> None:
        self.exploitable_bugs.append(exploit)
        self.safe = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "safe": self.safe,
            "violation_count": len(self.violations),
            "violations": [v.to_dict() for v in self.violations],
            "memory_leaks": [l.to_dict() for l in self.memory_leaks],
            "exploitable_bugs": [e.to_dict() for e in self.exploitable_bugs],
            "remediation": [{"type": r.violation_type.value,
                             "suggestion": r.suggestion,
                             "code_example": r.code_example,
                             "priority": r.priority}
                            for r in self.remediation],
            "stats": self.stats,
            "language": self.language,
        }


# ---------------------------------------------------------------------------
# Internal memory model
# ---------------------------------------------------------------------------

class AllocKind(Enum):
    STACK = auto()
    HEAP = auto()
    GLOBAL = auto()
    STATIC = auto()


class AllocLifetime(Enum):
    ALIVE = auto()
    FREED = auto()
    SCOPE_ENDED = auto()


@dataclass
class MemoryRegion:
    name: str
    kind: AllocKind
    size: int = 0
    alignment: int = 1
    lifetime: AllocLifetime = AllocLifetime.ALIVE
    allocated_at: str = ""
    freed_at: str = ""
    owner: str = ""
    initialized: bool = False
    accessible: bool = True
    written_by: Set[str] = field(default_factory=set)
    read_by: Set[str] = field(default_factory=set)

    def copy(self) -> "MemoryRegion":
        return MemoryRegion(
            name=self.name, kind=self.kind, size=self.size,
            alignment=self.alignment, lifetime=self.lifetime,
            allocated_at=self.allocated_at, freed_at=self.freed_at,
            owner=self.owner, initialized=self.initialized,
            accessible=self.accessible,
            written_by=set(self.written_by), read_by=set(self.read_by),
        )


@dataclass
class PointerState:
    name: str
    targets: Set[str] = field(default_factory=set)
    may_be_null: bool = True
    offset: int = 0
    type_name: str = ""
    is_stack_ptr: bool = False
    owner_scope_depth: int = 0

    def copy(self) -> "PointerState":
        return PointerState(
            name=self.name, targets=set(self.targets),
            may_be_null=self.may_be_null, offset=self.offset,
            type_name=self.type_name, is_stack_ptr=self.is_stack_ptr,
            owner_scope_depth=self.owner_scope_depth,
        )


@dataclass
class ResourceState:
    name: str
    resource_type: str = ""
    acquired: bool = False
    released: bool = False
    acquire_location: str = ""
    release_location: str = ""
    owner: str = ""
    drop_impl: bool = True

    def copy(self) -> "ResourceState":
        return ResourceState(
            name=self.name, resource_type=self.resource_type,
            acquired=self.acquired, released=self.released,
            acquire_location=self.acquire_location,
            release_location=self.release_location,
            owner=self.owner, drop_impl=self.drop_impl,
        )


@dataclass
class VerifierState:
    memory_regions: Dict[str, MemoryRegion] = field(default_factory=dict)
    pointers: Dict[str, PointerState] = field(default_factory=dict)
    resources: Dict[str, ResourceState] = field(default_factory=dict)
    scope_depth: int = 0
    scope_stack: List[Set[str]] = field(default_factory=list)
    stack_frame_size: int = 0
    max_stack_size: int = 8 * 1024 * 1024

    def copy(self) -> "VerifierState":
        return VerifierState(
            memory_regions={k: v.copy() for k, v in self.memory_regions.items()},
            pointers={k: v.copy() for k, v in self.pointers.items()},
            resources={k: v.copy() for k, v in self.resources.items()},
            scope_depth=self.scope_depth,
            scope_stack=[set(s) for s in self.scope_stack],
            stack_frame_size=self.stack_frame_size,
            max_stack_size=self.max_stack_size,
        )

    def push_scope(self) -> None:
        self.scope_depth += 1
        self.scope_stack.append(set())

    def pop_scope(self) -> Set[str]:
        if self.scope_stack:
            self.scope_depth -= 1
            return self.scope_stack.pop()
        return set()

    def add_to_scope(self, name: str) -> None:
        if self.scope_stack:
            self.scope_stack[-1].add(name)


# ---------------------------------------------------------------------------
# Memory Safety Verifier
# ---------------------------------------------------------------------------

class MemorySafetyVerifier:
    """Verify memory safety properties of C or Rust programs."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._state = VerifierState()
        self._report = SafetyReport()
        self._region_counter = 0
        self._CANARY_CHECK = self.config.get("stack_canary", True)

    def _reset(self, language: str = "c") -> None:
        self._state = VerifierState()
        self._report = SafetyReport(language=language)
        self._region_counter = 0

    def _new_region_name(self) -> str:
        self._region_counter += 1
        return f"region_{self._region_counter}"

    # --- spatial safety ---
    def _check_spatial_safety(self, ptr_name: str, access_offset: int,
                              access_size: int, loc: str) -> None:
        ptr = self._state.pointers.get(ptr_name)
        if not ptr:
            return

        if ptr.may_be_null:
            self._report.add_violation(SafetyViolation(
                type=SafetyViolationType.NULL_DEREF,
                location=loc, variable=ptr_name,
                description=f"Pointer `{ptr_name}` may be null when dereferenced",
                severity="error",
                suggestion=f"Add null check before dereferencing `{ptr_name}`",
            ))

        for target in ptr.targets:
            region = self._state.memory_regions.get(target)
            if not region:
                continue
            total_offset = ptr.offset + access_offset
            if total_offset < 0:
                self._report.add_violation(SafetyViolation(
                    type=SafetyViolationType.SPATIAL_NEGATIVE_INDEX,
                    location=loc, variable=ptr_name,
                    description=f"Negative offset {total_offset} on `{ptr_name}` into `{target}`",
                    severity="error",
                    exploitable=True,
                    exploit_difficulty=ExploitDifficulty.MODERATE,
                ))
            elif total_offset + access_size > region.size:
                vtype = (SafetyViolationType.STACK_BUFFER_OVERFLOW
                         if region.kind == AllocKind.STACK
                         else SafetyViolationType.HEAP_OVERFLOW)
                self._report.add_violation(SafetyViolation(
                    type=vtype,
                    location=loc, variable=ptr_name,
                    description=f"Access at offset {total_offset}+{access_size} exceeds `{target}` size {region.size}",
                    severity="error",
                    exploitable=True,
                    exploit_difficulty=ExploitDifficulty.EASY if region.kind == AllocKind.STACK else ExploitDifficulty.MODERATE,
                ))
                if region.kind == AllocKind.STACK and self._CANARY_CHECK:
                    self._report.add_violation(SafetyViolation(
                        type=SafetyViolationType.STACK_RETURN_OVERWRITE,
                        location=loc, variable=ptr_name,
                        description=f"Stack buffer overflow on `{target}` may overwrite return address",
                        severity="error",
                        exploitable=True,
                        exploit_difficulty=ExploitDifficulty.EASY,
                    ))

    # --- temporal safety ---
    def _check_temporal_safety(self, ptr_name: str, loc: str) -> None:
        ptr = self._state.pointers.get(ptr_name)
        if not ptr:
            return
        for target in ptr.targets:
            region = self._state.memory_regions.get(target)
            if not region:
                continue
            if region.lifetime == AllocLifetime.FREED:
                self._report.add_violation(SafetyViolation(
                    type=SafetyViolationType.TEMPORAL_USE_AFTER_FREE,
                    location=loc, variable=ptr_name,
                    description=f"Use-after-free: `{ptr_name}` references freed memory `{target}` (freed at {region.freed_at})",
                    severity="error",
                    exploitable=True,
                    exploit_difficulty=ExploitDifficulty.MODERATE,
                    suggestion="Set pointer to NULL after free",
                ))
            elif region.lifetime == AllocLifetime.SCOPE_ENDED:
                self._report.add_violation(SafetyViolation(
                    type=SafetyViolationType.TEMPORAL_DANGLING_PTR,
                    location=loc, variable=ptr_name,
                    description=f"Dangling pointer: `{ptr_name}` references stack variable `{target}` that has gone out of scope",
                    severity="error",
                    suggestion="Do not return pointers to local variables",
                ))

    # --- heap analysis ---
    def _process_alloc(self, var_name: str, size: int, loc: str) -> None:
        region_name = self._new_region_name()
        region = MemoryRegion(
            name=region_name, kind=AllocKind.HEAP, size=size,
            allocated_at=loc, owner=var_name,
        )
        self._state.memory_regions[region_name] = region
        self._state.pointers[var_name] = PointerState(
            name=var_name, targets={region_name}, may_be_null=True,
            type_name="void*",
        )

    def _process_free(self, ptr_name: str, loc: str) -> None:
        ptr = self._state.pointers.get(ptr_name)
        if not ptr:
            return
        for target in ptr.targets:
            region = self._state.memory_regions.get(target)
            if not region:
                continue
            if region.lifetime == AllocLifetime.FREED:
                self._report.add_violation(SafetyViolation(
                    type=SafetyViolationType.TEMPORAL_DOUBLE_FREE,
                    location=loc, variable=ptr_name,
                    description=f"Double free of `{ptr_name}` (previously freed at {region.freed_at})",
                    severity="error",
                    exploitable=True,
                    exploit_difficulty=ExploitDifficulty.MODERATE,
                ))
            else:
                region.lifetime = AllocLifetime.FREED
                region.freed_at = loc

    def _check_heap_leaks(self) -> None:
        for name, region in self._state.memory_regions.items():
            if region.kind == AllocKind.HEAP and region.lifetime == AllocLifetime.ALIVE:
                still_pointed = any(
                    name in ptr.targets
                    for ptr in self._state.pointers.values()
                )
                if not still_pointed:
                    self._report.add_leak(MemoryLeak(
                        allocation_site=region.allocated_at,
                        variable=name,
                        size=region.size,
                        reachable=False,
                        leak_path=f"Allocated at {region.allocated_at}, no remaining pointers",
                    ))

    # --- stack safety ---
    def _process_stack_alloc(self, var_name: str, size: int, loc: str) -> None:
        region_name = f"stack_{var_name}"
        region = MemoryRegion(
            name=region_name, kind=AllocKind.STACK, size=size,
            allocated_at=loc, owner=var_name, initialized=False,
        )
        self._state.memory_regions[region_name] = region
        self._state.pointers[var_name] = PointerState(
            name=var_name, targets={region_name}, may_be_null=False,
            is_stack_ptr=True,
            owner_scope_depth=self._state.scope_depth,
        )
        self._state.stack_frame_size += size
        self._state.add_to_scope(var_name)

        if self._state.stack_frame_size > self._state.max_stack_size:
            self._report.add_violation(SafetyViolation(
                type=SafetyViolationType.STACK_UNDERFLOW,
                location=loc, variable=var_name,
                description=f"Stack frame size {self._state.stack_frame_size} exceeds limit {self._state.max_stack_size}",
                severity="error",
            ))

    # --- aliasing analysis ---
    def _check_alias_conflict(self, writer_ptr: str, loc: str) -> None:
        writer_state = self._state.pointers.get(writer_ptr)
        if not writer_state:
            return
        for other_name, other_ptr in self._state.pointers.items():
            if other_name == writer_ptr:
                continue
            shared_targets = writer_state.targets & other_ptr.targets
            if shared_targets:
                for target in shared_targets:
                    region = self._state.memory_regions.get(target)
                    if region and other_name in region.read_by:
                        self._report.add_violation(SafetyViolation(
                            type=SafetyViolationType.ALIAS_CONFLICT,
                            location=loc, variable=writer_ptr,
                            description=f"Conflicting write through `{writer_ptr}` while `{other_name}` has read access to `{target}`",
                            severity="warning",
                            suggestion="Use restrict qualifier (C) or ensure exclusive access (Rust)",
                        ))

    # --- RAII verification (Rust) ---
    def _process_resource_acquire(self, var_name: str, resource_type: str,
                                  loc: str) -> None:
        self._state.resources[var_name] = ResourceState(
            name=var_name, resource_type=resource_type,
            acquired=True, acquire_location=loc,
            owner=var_name,
        )
        self._state.add_to_scope(var_name)

    def _process_resource_release(self, var_name: str, loc: str) -> None:
        res = self._state.resources.get(var_name)
        if not res:
            return
        if res.released:
            self._report.add_violation(SafetyViolation(
                type=SafetyViolationType.RAII_DOUBLE_DROP,
                location=loc, variable=var_name,
                description=f"Double drop of resource `{var_name}` (first at {res.release_location})",
                severity="error",
            ))
        else:
            res.released = True
            res.release_location = loc

    def _check_raii_leaks(self) -> None:
        for name, res in self._state.resources.items():
            if res.acquired and not res.released and res.drop_impl:
                self._report.add_violation(SafetyViolation(
                    type=SafetyViolationType.RAII_RESOURCE_LEAK,
                    location=res.acquire_location, variable=name,
                    description=f"Resource `{name}` ({res.resource_type}) acquired but never released",
                    severity="warning",
                    suggestion="Ensure the resource is properly dropped or explicitly released",
                ))

    # --- exploit detection ---
    def _assess_exploitability(self) -> None:
        for v in self._report.violations:
            if not v.exploitable:
                continue
            exploit = None
            if v.type == SafetyViolationType.STACK_BUFFER_OVERFLOW:
                exploit = ExploitableBug(
                    violation=v,
                    exploit_type="stack-based buffer overflow → RCE",
                    difficulty=ExploitDifficulty.EASY,
                    description="Stack buffer overflow can overwrite return address for code execution",
                    cwe_id="CWE-121",
                )
            elif v.type == SafetyViolationType.HEAP_OVERFLOW:
                exploit = ExploitableBug(
                    violation=v,
                    exploit_type="heap overflow → heap corruption",
                    difficulty=ExploitDifficulty.MODERATE,
                    description="Heap buffer overflow can corrupt heap metadata for arbitrary write",
                    cwe_id="CWE-122",
                )
            elif v.type == SafetyViolationType.TEMPORAL_USE_AFTER_FREE:
                exploit = ExploitableBug(
                    violation=v,
                    exploit_type="use-after-free → type confusion",
                    difficulty=ExploitDifficulty.MODERATE,
                    description="Use-after-free can lead to type confusion if allocation is reused",
                    cwe_id="CWE-416",
                )
            elif v.type == SafetyViolationType.TEMPORAL_DOUBLE_FREE:
                exploit = ExploitableBug(
                    violation=v,
                    exploit_type="double-free → heap corruption",
                    difficulty=ExploitDifficulty.MODERATE,
                    description="Double free corrupts heap allocator state",
                    cwe_id="CWE-415",
                )
            elif v.type == SafetyViolationType.STACK_RETURN_OVERWRITE:
                exploit = ExploitableBug(
                    violation=v,
                    exploit_type="return address overwrite → RCE",
                    difficulty=ExploitDifficulty.EASY,
                    description="Overwriting return address allows arbitrary code execution",
                    cwe_id="CWE-121",
                )
            if exploit:
                self._report.add_exploit(exploit)

    # --- remediation ---
    def _generate_remediation(self) -> None:
        seen_types: Set[SafetyViolationType] = set()
        for v in self._report.violations:
            if v.type in seen_types:
                continue
            seen_types.add(v.type)

            remediation_map = {
                SafetyViolationType.SPATIAL_OUT_OF_BOUNDS: Remediation(
                    violation_type=v.type,
                    suggestion="Add bounds checking before array access",
                    code_example="if (index >= 0 && index < size) { arr[index]; }",
                    priority="high",
                ),
                SafetyViolationType.TEMPORAL_USE_AFTER_FREE: Remediation(
                    violation_type=v.type,
                    suggestion="Set pointer to NULL after free; use smart pointers in C++",
                    code_example="free(ptr); ptr = NULL;",
                    priority="high",
                ),
                SafetyViolationType.HEAP_LEAK: Remediation(
                    violation_type=v.type,
                    suggestion="Ensure all malloc'd memory is freed; consider RAII pattern",
                    code_example="// Ensure matching free() for every malloc()",
                    priority="medium",
                ),
                SafetyViolationType.STACK_BUFFER_OVERFLOW: Remediation(
                    violation_type=v.type,
                    suggestion="Use bounded string functions (strncpy, snprintf)",
                    code_example="strncpy(dst, src, sizeof(dst) - 1); dst[sizeof(dst)-1] = '\\0';",
                    priority="critical",
                ),
                SafetyViolationType.NULL_DEREF: Remediation(
                    violation_type=v.type,
                    suggestion="Check pointer for NULL before dereferencing",
                    code_example="if (ptr != NULL) { *ptr = value; }",
                    priority="high",
                ),
                SafetyViolationType.ALIAS_CONFLICT: Remediation(
                    violation_type=v.type,
                    suggestion="Use restrict qualifier to indicate no aliasing",
                    code_example="void func(int * restrict a, int * restrict b)",
                    priority="medium",
                ),
            }

            rem = remediation_map.get(v.type)
            if rem:
                self._report.remediation.append(rem)
            else:
                self._report.remediation.append(Remediation(
                    violation_type=v.type,
                    suggestion=v.suggestion or f"Fix {v.type.value} violation",
                    priority="medium",
                ))

    # --- statement analysis ---
    def _analyze_stmt(self, stmt: Dict[str, Any], language: str) -> None:
        kind = stmt.get("kind", "")
        loc = stmt.get("location", "unknown")

        if kind == "decl":
            name = stmt.get("name", "")
            var_type = stmt.get("type", "int")
            size = stmt.get("size", 4)
            if var_type.endswith("*"):
                self._state.pointers[name] = PointerState(
                    name=name, may_be_null=True, type_name=var_type)
            else:
                self._process_stack_alloc(name, size, loc)
            if "init" in stmt:
                init = stmt["init"]
                if isinstance(init, dict):
                    if init.get("kind") == "call" and init.get("func") in ("malloc", "calloc"):
                        alloc_size = init.get("size", 0)
                        self._process_alloc(name, alloc_size, loc)
                    elif init.get("kind") == "const" and init.get("value") == 0:
                        if name in self._state.pointers:
                            self._state.pointers[name].may_be_null = True

        elif kind == "assign":
            target = stmt.get("target", "")
            value = stmt.get("value")
            if isinstance(value, dict):
                if value.get("kind") == "call" and value.get("func") in ("malloc", "calloc"):
                    alloc_size = value.get("size", 0)
                    self._process_alloc(target, alloc_size, loc)
                elif value.get("kind") == "addr":
                    src = value.get("operand", {}).get("name", "")
                    if src:
                        src_region = f"stack_{src}"
                        self._state.pointers[target] = PointerState(
                            name=target, targets={src_region},
                            may_be_null=False, is_stack_ptr=True,
                            owner_scope_depth=self._state.scope_depth,
                        )

        elif kind == "deref":
            ptr_name = stmt.get("pointer", "")
            access_size = stmt.get("access_size", 1)
            offset = stmt.get("offset", 0)
            self._check_temporal_safety(ptr_name, loc)
            self._check_spatial_safety(ptr_name, offset, access_size, loc)

        elif kind == "index":
            base = stmt.get("base", "")
            index = stmt.get("index", 0)
            elem_size = stmt.get("elem_size", 1)
            self._check_temporal_safety(base, loc)
            self._check_spatial_safety(base, index * elem_size, elem_size, loc)

        elif kind == "free":
            ptr_name = stmt.get("pointer", "")
            self._process_free(ptr_name, loc)

        elif kind == "write":
            ptr_name = stmt.get("pointer", "")
            access_size = stmt.get("access_size", 1)
            offset = stmt.get("offset", 0)
            self._check_temporal_safety(ptr_name, loc)
            self._check_spatial_safety(ptr_name, offset, access_size, loc)
            self._check_alias_conflict(ptr_name, loc)
            for target in self._state.pointers.get(ptr_name, PointerState(name="")).targets:
                region = self._state.memory_regions.get(target)
                if region:
                    region.written_by.add(ptr_name)

        elif kind == "read":
            ptr_name = stmt.get("pointer", "")
            self._check_temporal_safety(ptr_name, loc)
            for target in self._state.pointers.get(ptr_name, PointerState(name="")).targets:
                region = self._state.memory_regions.get(target)
                if region:
                    region.read_by.add(ptr_name)
                    if not region.initialized:
                        self._report.add_violation(SafetyViolation(
                            type=SafetyViolationType.UNINIT_READ,
                            location=loc, variable=ptr_name,
                            description=f"Reading from uninitialized memory `{target}` through `{ptr_name}`",
                            severity="warning",
                        ))

        elif kind == "resource_acquire":
            var_name = stmt.get("name", "")
            res_type = stmt.get("resource_type", "")
            self._process_resource_acquire(var_name, res_type, loc)

        elif kind == "resource_release":
            var_name = stmt.get("name", "")
            self._process_resource_release(var_name, loc)

        elif kind == "block":
            self._state.push_scope()
            for s in stmt.get("body", []):
                self._analyze_stmt(s, language)
            leaving = self._state.pop_scope()
            for var in leaving:
                region_name = f"stack_{var}"
                region = self._state.memory_regions.get(region_name)
                if region:
                    region.lifetime = AllocLifetime.SCOPE_ENDED
                if language == "rust":
                    res = self._state.resources.get(var)
                    if res and res.acquired and not res.released:
                        res.released = True
                        res.release_location = f"scope_exit({loc})"

        elif kind == "if":
            saved = self._state.copy()
            for s in stmt.get("then", []):
                self._analyze_stmt(s, language)
            then_state = self._state
            self._state = saved
            for s in stmt.get("else", []):
                self._analyze_stmt(s, language)
            # join: use then_state violations
            self._report.violations.extend(
                v for v in then_state.memory_regions.values()
                if False  # dummy, we already reported
            )

        elif kind in ("while", "for"):
            for _ in range(2):
                for s in stmt.get("body", []):
                    self._analyze_stmt(s, language)

        elif kind == "return":
            ret_val = stmt.get("value")
            if isinstance(ret_val, dict) and ret_val.get("kind") == "addr":
                var = ret_val.get("operand", {}).get("name", "")
                if var:
                    region = self._state.memory_regions.get(f"stack_{var}")
                    if region and region.kind == AllocKind.STACK:
                        self._report.add_violation(SafetyViolation(
                            type=SafetyViolationType.TEMPORAL_DANGLING_PTR,
                            location=loc, variable=var,
                            description=f"Returning pointer to local variable `{var}`",
                            severity="error",
                            suggestion="Allocate on heap instead",
                        ))

        elif kind == "call":
            func = stmt.get("func", "")
            args = stmt.get("args", [])
            if func == "free" and args:
                if isinstance(args[0], dict) and args[0].get("kind") == "var":
                    self._process_free(args[0]["name"], loc)

    # --- main entry ---
    def verify(self, program: Any, language: str = "c") -> SafetyReport:
        self._reset(language)

        if isinstance(program, dict):
            stmts = program.get("body", [])
            params = program.get("params", [])
        elif isinstance(program, list):
            stmts = program
            params = []
        else:
            return self._report

        self._state.push_scope()

        for param in params:
            name = param.get("name", "")
            ptype = param.get("type", "int")
            if ptype.endswith("*"):
                self._state.pointers[name] = PointerState(
                    name=name, may_be_null=True, type_name=ptype)
            else:
                size = param.get("size", 4)
                self._process_stack_alloc(name, size, "param")

        for stmt in stmts:
            self._analyze_stmt(stmt, language)

        self._state.pop_scope()

        self._check_heap_leaks()
        if language == "rust":
            self._check_raii_leaks()
        self._assess_exploitability()
        self._generate_remediation()

        self._report.stats = {
            "total_violations": len(self._report.violations),
            "memory_leaks": len(self._report.memory_leaks),
            "exploitable_bugs": len(self._report.exploitable_bugs),
            "regions_tracked": len(self._state.memory_regions),
            "pointers_tracked": len(self._state.pointers),
        }

        return self._report

    def verify_c(self, program: Any) -> SafetyReport:
        return self.verify(program, "c")

    def verify_rust(self, program: Any) -> SafetyReport:
        return self.verify(program, "rust")
