"""
Simplified Rust borrow checker.
Implements ownership tracking, move semantics, borrow tracking,
borrow rules, lifetime analysis, drop order analysis, and
pattern matching borrow tracking.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from enum import Enum, auto
import copy


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Mutability(Enum):
    IMMUTABLE = auto()
    MUTABLE = auto()


class OwnershipState(Enum):
    OWNED = auto()
    MOVED = auto()
    BORROWED_SHARED = auto()
    BORROWED_MUT = auto()
    DROPPED = auto()
    PARTIALLY_MOVED = auto()


class BorrowKind(Enum):
    SHARED = auto()     # &T
    MUTABLE = auto()    # &mut T


class ViolationSeverity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Lifetime:
    name: str
    scope_depth: int
    start_point: int = 0
    end_point: int = 0
    outlives: List[str] = field(default_factory=list)

    def contains(self, point: int) -> bool:
        return self.start_point <= point <= self.end_point

    def outlives_lt(self, other: "Lifetime") -> bool:
        return self.scope_depth <= other.scope_depth


@dataclass
class BorrowInfo:
    borrow_id: int
    kind: BorrowKind
    borrowed_place: str
    borrower: str
    lifetime: Lifetime
    location: str = ""
    is_active: bool = True
    reborrow_of: Optional[int] = None

    def conflicts_with(self, other: "BorrowInfo") -> bool:
        if not self.is_active or not other.is_active:
            return False
        if self.borrowed_place != other.borrowed_place:
            if not (self.borrowed_place.startswith(other.borrowed_place + ".") or
                    other.borrowed_place.startswith(self.borrowed_place + ".")):
                return False
        if self.kind == BorrowKind.MUTABLE or other.kind == BorrowKind.MUTABLE:
            return True
        return False


@dataclass
class OwnershipInfo:
    variable: str
    state: OwnershipState = OwnershipState.OWNED
    move_location: Optional[str] = None
    drop_order: int = -1
    type_name: str = ""
    is_copy: bool = False
    fields: Dict[str, "OwnershipInfo"] = field(default_factory=dict)
    lifetime: Optional[Lifetime] = None


@dataclass
class BorrowViolation:
    location: str
    description: str
    severity: ViolationSeverity = ViolationSeverity.ERROR
    violated_rule: str = ""
    suggestion: str = ""
    related_locations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "location": self.location,
            "description": self.description,
            "severity": self.severity.value,
            "violated_rule": self.violated_rule,
            "suggestion": self.suggestion,
            "related_locations": self.related_locations,
        }


@dataclass
class BorrowCheckResult:
    valid: bool = True
    violations: List[BorrowViolation] = field(default_factory=list)
    ownership_map: Dict[str, OwnershipInfo] = field(default_factory=dict)
    active_borrows: List[BorrowInfo] = field(default_factory=list)
    drop_order: List[str] = field(default_factory=list)
    lifetime_errors: List[str] = field(default_factory=list)

    def add_violation(self, violation: BorrowViolation) -> None:
        self.violations.append(violation)
        if violation.severity == ViolationSeverity.ERROR:
            self.valid = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "violation_count": len(self.violations),
            "violations": [v.to_dict() for v in self.violations],
            "drop_order": self.drop_order,
            "lifetime_errors": self.lifetime_errors,
        }


# ---------------------------------------------------------------------------
# Scope tracking
# ---------------------------------------------------------------------------

@dataclass
class Scope:
    depth: int
    variables: List[str] = field(default_factory=list)
    borrows: List[int] = field(default_factory=list)
    parent: Optional["Scope"] = None

    def add_variable(self, var: str) -> None:
        self.variables.append(var)

    def add_borrow(self, borrow_id: int) -> None:
        self.borrows.append(borrow_id)


# ---------------------------------------------------------------------------
# Type system helpers
# ---------------------------------------------------------------------------

COPY_TYPES = frozenset({
    "i8", "i16", "i32", "i64", "i128", "isize",
    "u8", "u16", "u32", "u64", "u128", "usize",
    "f32", "f64", "bool", "char",
    "()", "&str",
})


def is_copy_type(type_name: str) -> bool:
    if type_name in COPY_TYPES:
        return True
    if type_name.startswith("&") and not type_name.startswith("&mut"):
        return True
    if type_name.startswith("("):
        inner = type_name[1:-1]
        parts = [p.strip() for p in inner.split(",") if p.strip()]
        return all(is_copy_type(p) for p in parts)
    if type_name.startswith("[") and ";" in type_name:
        elem_type = type_name[1:type_name.index(";")].strip()
        return is_copy_type(elem_type)
    return False


def type_has_drop(type_name: str) -> bool:
    if is_copy_type(type_name):
        return False
    non_drop = {"&str", "()", "bool"}
    if type_name in non_drop:
        return False
    return True


# ---------------------------------------------------------------------------
# Borrow Checker
# ---------------------------------------------------------------------------

class BorrowChecker:
    """Simplified Rust borrow checker."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._borrow_id_counter = 0
        self._program_point = 0
        self._scope_depth = 0
        self._scopes: List[Scope] = []
        self._ownership: Dict[str, OwnershipInfo] = {}
        self._borrows: Dict[int, BorrowInfo] = {}
        self._active_borrows_for_place: Dict[str, List[int]] = {}
        self._result = BorrowCheckResult()
        self._drop_counter = 0
        self._lifetimes: Dict[str, Lifetime] = {}

    def _reset(self) -> None:
        self._borrow_id_counter = 0
        self._program_point = 0
        self._scope_depth = 0
        self._scopes = []
        self._ownership = {}
        self._borrows = {}
        self._active_borrows_for_place = {}
        self._result = BorrowCheckResult()
        self._drop_counter = 0
        self._lifetimes = {}

    def _next_borrow_id(self) -> int:
        self._borrow_id_counter += 1
        return self._borrow_id_counter

    def _advance_point(self) -> int:
        self._program_point += 1
        return self._program_point

    # --- scope management ---
    def _enter_scope(self) -> Scope:
        self._scope_depth += 1
        scope = Scope(depth=self._scope_depth)
        if self._scopes:
            scope.parent = self._scopes[-1]
        self._scopes.append(scope)
        return scope

    def _exit_scope(self) -> None:
        if not self._scopes:
            return
        scope = self._scopes.pop()
        self._scope_depth -= 1

        for var in reversed(scope.variables):
            self._drop_variable(var)

        for bid in scope.borrows:
            if bid in self._borrows:
                self._borrows[bid].is_active = False
                place = self._borrows[bid].borrowed_place
                if place in self._active_borrows_for_place:
                    self._active_borrows_for_place[place] = [
                        b for b in self._active_borrows_for_place[place]
                        if b != bid
                    ]

    def _drop_variable(self, var: str) -> None:
        info = self._ownership.get(var)
        if not info:
            return
        if info.state == OwnershipState.OWNED:
            active = self._get_active_borrows(var)
            if active:
                for bid in active:
                    borrow = self._borrows[bid]
                    self._result.add_violation(BorrowViolation(
                        location=f"drop({var})",
                        description=f"Cannot drop `{var}` while it is borrowed by `{borrow.borrower}`",
                        severity=ViolationSeverity.ERROR,
                        violated_rule="E0505: cannot move out of borrowed content",
                        suggestion=f"Ensure borrows of `{var}` end before it goes out of scope",
                        related_locations=[borrow.location],
                    ))
            info.state = OwnershipState.DROPPED
            self._drop_counter += 1
            info.drop_order = self._drop_counter
            self._result.drop_order.append(var)

            for field_name, field_info in info.fields.items():
                if field_info.state == OwnershipState.OWNED:
                    field_info.state = OwnershipState.DROPPED
                    self._drop_counter += 1
                    field_info.drop_order = self._drop_counter

    # --- ownership operations ---
    def _declare_variable(self, var: str, type_name: str,
                          location: str) -> None:
        lt = Lifetime(
            name=f"'{var}",
            scope_depth=self._scope_depth,
            start_point=self._program_point,
            end_point=self._program_point + 1000,
        )
        self._lifetimes[f"'{var}"] = lt

        info = OwnershipInfo(
            variable=var,
            state=OwnershipState.OWNED,
            type_name=type_name,
            is_copy=is_copy_type(type_name),
            lifetime=lt,
        )
        self._ownership[var] = info

        if self._scopes:
            self._scopes[-1].add_variable(var)

    def _move_variable(self, src: str, dst: str, location: str) -> bool:
        src_info = self._ownership.get(src)
        if not src_info:
            self._result.add_violation(BorrowViolation(
                location=location,
                description=f"Use of undeclared variable `{src}`",
                severity=ViolationSeverity.ERROR,
                violated_rule="E0425: cannot find value",
            ))
            return False

        if src_info.is_copy:
            dst_info = OwnershipInfo(
                variable=dst,
                state=OwnershipState.OWNED,
                type_name=src_info.type_name,
                is_copy=True,
            )
            self._ownership[dst] = dst_info
            if self._scopes:
                self._scopes[-1].add_variable(dst)
            return True

        if src_info.state == OwnershipState.MOVED:
            self._result.add_violation(BorrowViolation(
                location=location,
                description=f"Use of moved value `{src}` (moved at {src_info.move_location})",
                severity=ViolationSeverity.ERROR,
                violated_rule="E0382: use of moved value",
                suggestion=f"Consider cloning `{src}` before the first move",
                related_locations=[src_info.move_location or "unknown"],
            ))
            return False

        if src_info.state == OwnershipState.DROPPED:
            self._result.add_violation(BorrowViolation(
                location=location,
                description=f"Use of dropped value `{src}`",
                severity=ViolationSeverity.ERROR,
                violated_rule="E0382: use of dropped value",
            ))
            return False

        active = self._get_active_borrows(src)
        if active:
            borrow = self._borrows[active[0]]
            self._result.add_violation(BorrowViolation(
                location=location,
                description=f"Cannot move `{src}` while borrowed by `{borrow.borrower}`",
                severity=ViolationSeverity.ERROR,
                violated_rule="E0505: cannot move out of borrowed content",
                suggestion=f"Ensure borrows of `{src}` end before moving",
                related_locations=[borrow.location],
            ))
            return False

        src_info.state = OwnershipState.MOVED
        src_info.move_location = location

        dst_info = OwnershipInfo(
            variable=dst,
            state=OwnershipState.OWNED,
            type_name=src_info.type_name,
            is_copy=src_info.is_copy,
            fields=copy.deepcopy(src_info.fields),
        )
        self._ownership[dst] = dst_info
        if self._scopes:
            self._scopes[-1].add_variable(dst)
        return True

    def _use_variable(self, var: str, location: str) -> bool:
        info = self._ownership.get(var)
        if not info:
            self._result.add_violation(BorrowViolation(
                location=location,
                description=f"Use of undeclared variable `{var}`",
                severity=ViolationSeverity.ERROR,
                violated_rule="E0425: cannot find value",
            ))
            return False

        if info.state == OwnershipState.MOVED:
            self._result.add_violation(BorrowViolation(
                location=location,
                description=f"Use of moved value `{var}` (moved at {info.move_location})",
                severity=ViolationSeverity.ERROR,
                violated_rule="E0382: use of moved value",
                suggestion=f"Consider using a reference or clone",
                related_locations=[info.move_location or "unknown"],
            ))
            return False

        if info.state == OwnershipState.DROPPED:
            self._result.add_violation(BorrowViolation(
                location=location,
                description=f"Use of dropped value `{var}`",
                severity=ViolationSeverity.ERROR,
                violated_rule="E0382: use after drop",
            ))
            return False

        return True

    # --- borrowing ---
    def _create_borrow(self, borrower: str, place: str,
                       kind: BorrowKind, location: str,
                       lifetime_name: Optional[str] = None) -> Optional[int]:
        info = self._ownership.get(place)
        if not info:
            self._result.add_violation(BorrowViolation(
                location=location,
                description=f"Cannot borrow undeclared variable `{place}`",
                severity=ViolationSeverity.ERROR,
                violated_rule="E0425: cannot find value",
            ))
            return None

        if info.state == OwnershipState.MOVED:
            self._result.add_violation(BorrowViolation(
                location=location,
                description=f"Cannot borrow moved value `{place}`",
                severity=ViolationSeverity.ERROR,
                violated_rule="E0382: borrow of moved value",
                related_locations=[info.move_location or "unknown"],
            ))
            return None

        active = self._get_active_borrows(place)
        for bid in active:
            existing = self._borrows[bid]
            if kind == BorrowKind.MUTABLE:
                if existing.kind == BorrowKind.SHARED:
                    self._result.add_violation(BorrowViolation(
                        location=location,
                        description=f"Cannot borrow `{place}` as mutable because it is also borrowed as immutable by `{existing.borrower}`",
                        severity=ViolationSeverity.ERROR,
                        violated_rule="E0502: cannot borrow as mutable because also borrowed as immutable",
                        suggestion="Ensure shared borrows end before taking a mutable borrow",
                        related_locations=[existing.location],
                    ))
                    return None
                elif existing.kind == BorrowKind.MUTABLE:
                    self._result.add_violation(BorrowViolation(
                        location=location,
                        description=f"Cannot borrow `{place}` as mutable more than once: already borrowed by `{existing.borrower}`",
                        severity=ViolationSeverity.ERROR,
                        violated_rule="E0499: cannot borrow as mutable more than once",
                        suggestion="Use a block to limit the scope of the first mutable borrow",
                        related_locations=[existing.location],
                    ))
                    return None
            elif kind == BorrowKind.SHARED:
                if existing.kind == BorrowKind.MUTABLE:
                    self._result.add_violation(BorrowViolation(
                        location=location,
                        description=f"Cannot borrow `{place}` as immutable because it is also borrowed as mutable by `{existing.borrower}`",
                        severity=ViolationSeverity.ERROR,
                        violated_rule="E0502: cannot borrow as immutable because also borrowed as mutable",
                        related_locations=[existing.location],
                    ))
                    return None

        lt = Lifetime(
            name=lifetime_name or f"'borrow_{self._borrow_id_counter + 1}",
            scope_depth=self._scope_depth,
            start_point=self._program_point,
            end_point=self._program_point + 100,
        )

        bid = self._next_borrow_id()
        borrow = BorrowInfo(
            borrow_id=bid, kind=kind, borrowed_place=place,
            borrower=borrower, lifetime=lt, location=location,
        )
        self._borrows[bid] = borrow

        if place not in self._active_borrows_for_place:
            self._active_borrows_for_place[place] = []
        self._active_borrows_for_place[place].append(bid)

        if self._scopes:
            self._scopes[-1].add_borrow(bid)

        ref_type = f"&{place}" if kind == BorrowKind.SHARED else f"&mut {place}"
        self._declare_variable(borrower, ref_type, location)

        return bid

    def _end_borrow(self, borrow_id: int) -> None:
        if borrow_id in self._borrows:
            borrow = self._borrows[borrow_id]
            borrow.is_active = False
            borrow.lifetime.end_point = self._program_point
            place = borrow.borrowed_place
            if place in self._active_borrows_for_place:
                self._active_borrows_for_place[place] = [
                    b for b in self._active_borrows_for_place[place]
                    if b != borrow_id
                ]

    def _get_active_borrows(self, place: str) -> List[int]:
        result = []
        for p, bids in self._active_borrows_for_place.items():
            if p == place or p.startswith(place + ".") or place.startswith(p + "."):
                for bid in bids:
                    if bid in self._borrows and self._borrows[bid].is_active:
                        result.append(bid)
        return result

    def _write_through_ref(self, ref_var: str, location: str) -> bool:
        info = self._ownership.get(ref_var)
        if not info:
            return False
        if not info.type_name.startswith("&mut"):
            self._result.add_violation(BorrowViolation(
                location=location,
                description=f"Cannot write through shared reference `{ref_var}`",
                severity=ViolationSeverity.ERROR,
                violated_rule="E0594: cannot assign to immutable borrow",
            ))
            return False
        return True

    # --- lifetime analysis ---
    def _check_lifetime_validity(self, ref_var: str, referent: str,
                                 location: str) -> bool:
        ref_info = self._ownership.get(ref_var)
        referent_info = self._ownership.get(referent)

        if not ref_info or not referent_info:
            return True

        ref_lt = ref_info.lifetime
        referent_lt = referent_info.lifetime

        if ref_lt and referent_lt:
            if not referent_lt.outlives_lt(ref_lt):
                self._result.add_violation(BorrowViolation(
                    location=location,
                    description=f"`{referent}` does not live long enough: borrowed by `{ref_var}` which has a longer lifetime",
                    severity=ViolationSeverity.ERROR,
                    violated_rule="E0597: does not live long enough",
                    suggestion=f"Ensure `{referent}` lives at least as long as `{ref_var}`",
                ))
                self._result.lifetime_errors.append(
                    f"{referent} does not live long enough for {ref_var}")
                return False
        return True

    def _check_return_lifetime(self, returned_ref: str,
                               location: str) -> bool:
        info = self._ownership.get(returned_ref)
        if not info:
            return True
        if info.type_name.startswith("&"):
            if info.lifetime and info.lifetime.scope_depth > 0:
                for bid, borrow in self._borrows.items():
                    if borrow.borrower == returned_ref and borrow.is_active:
                        referent_info = self._ownership.get(borrow.borrowed_place)
                        if referent_info and referent_info.lifetime:
                            if referent_info.lifetime.scope_depth >= self._scope_depth:
                                self._result.add_violation(BorrowViolation(
                                    location=location,
                                    description=f"Cannot return reference `{returned_ref}` to local variable `{borrow.borrowed_place}`",
                                    severity=ViolationSeverity.ERROR,
                                    violated_rule="E0515: cannot return reference to local variable",
                                    suggestion="Consider returning an owned value instead",
                                ))
                                return False
        return True

    # --- pattern matching ---
    def _check_match_borrows(self, scrutinee: str, arms: List[Dict[str, Any]],
                             location: str) -> None:
        scrutinee_info = self._ownership.get(scrutinee)
        if not scrutinee_info:
            return

        for arm in arms:
            pattern = arm.get("pattern", {})
            binding_mode = pattern.get("binding_mode", "move")
            bindings = pattern.get("bindings", [])

            for binding in bindings:
                bind_name = binding.get("name", "")
                bind_field = binding.get("field", scrutinee)

                if binding_mode == "ref":
                    self._create_borrow(
                        bind_name, bind_field, BorrowKind.SHARED, location)
                elif binding_mode == "ref_mut":
                    self._create_borrow(
                        bind_name, bind_field, BorrowKind.MUTABLE, location)
                elif binding_mode == "move":
                    if not scrutinee_info.is_copy:
                        partial_place = f"{scrutinee}.{bind_field}" if bind_field != scrutinee else scrutinee
                        self._move_variable(partial_place, bind_name, location)
                    else:
                        self._declare_variable(bind_name, scrutinee_info.type_name, location)

    def _check_if_let_borrow(self, scrutinee: str, pattern: Dict[str, Any],
                             location: str) -> None:
        binding_mode = pattern.get("binding_mode", "move")
        bindings = pattern.get("bindings", [])

        for binding in bindings:
            bind_name = binding.get("name", "")
            bind_field = binding.get("field", scrutinee)
            if binding_mode == "ref":
                self._create_borrow(bind_name, bind_field, BorrowKind.SHARED, location)
            elif binding_mode == "ref_mut":
                self._create_borrow(bind_name, bind_field, BorrowKind.MUTABLE, location)

    # --- statement processing ---
    def _process_stmt(self, stmt: Dict[str, Any]) -> None:
        self._advance_point()
        kind = stmt.get("kind", "")
        loc = stmt.get("location", f"line:{self._program_point}")

        if kind == "let":
            var = stmt.get("name", "")
            type_name = stmt.get("type", "unknown")
            self._declare_variable(var, type_name, loc)

            init = stmt.get("init")
            if init:
                init_kind = init.get("kind", "")
                if init_kind == "move":
                    src = init.get("source", "")
                    self._move_variable(src, var, loc)
                elif init_kind == "borrow":
                    place = init.get("place", "")
                    borrow_kind = BorrowKind.MUTABLE if init.get("mutable") else BorrowKind.SHARED
                    self._create_borrow(var, place, borrow_kind, loc)
                elif init_kind == "call":
                    for arg in init.get("args", []):
                        if isinstance(arg, dict):
                            if arg.get("kind") == "move":
                                self._move_variable(arg["source"], f"_arg_{self._program_point}", loc)
                            elif arg.get("kind") == "borrow":
                                bk = BorrowKind.MUTABLE if arg.get("mutable") else BorrowKind.SHARED
                                self._create_borrow(
                                    f"_arg_{self._program_point}",
                                    arg["place"], bk, loc)

        elif kind == "assign":
            target = stmt.get("target", "")
            value = stmt.get("value", {})
            self._use_variable(target, loc)

            if isinstance(value, dict):
                vk = value.get("kind", "")
                if vk == "move":
                    src = value.get("source", "")
                    info = self._ownership.get(target)
                    if info and info.state == OwnershipState.OWNED and type_has_drop(info.type_name):
                        self._drop_variable(target)
                    self._move_variable(src, target, loc)
                elif vk == "borrow":
                    place = value.get("place", "")
                    bk = BorrowKind.MUTABLE if value.get("mutable") else BorrowKind.SHARED
                    self._create_borrow(target, place, bk, loc)

        elif kind == "use":
            var = stmt.get("name", "")
            self._use_variable(var, loc)

        elif kind == "write_ref":
            ref_var = stmt.get("ref", "")
            self._write_through_ref(ref_var, loc)

        elif kind == "drop":
            var = stmt.get("name", "")
            self._drop_variable(var)

        elif kind == "move":
            src = stmt.get("source", "")
            dst = stmt.get("dest", "")
            self._move_variable(src, dst, loc)

        elif kind == "borrow":
            borrower = stmt.get("borrower", "")
            place = stmt.get("place", "")
            bk = BorrowKind.MUTABLE if stmt.get("mutable") else BorrowKind.SHARED
            self._create_borrow(borrower, place, bk, loc)

        elif kind == "end_borrow":
            borrow_id = stmt.get("borrow_id")
            if borrow_id is not None:
                self._end_borrow(borrow_id)

        elif kind == "block":
            self._enter_scope()
            for s in stmt.get("body", []):
                self._process_stmt(s)
            self._exit_scope()

        elif kind == "if":
            cond_var = stmt.get("cond_var")
            if cond_var:
                self._use_variable(cond_var, loc)
            self._enter_scope()
            for s in stmt.get("then", []):
                self._process_stmt(s)
            self._exit_scope()
            if stmt.get("else"):
                self._enter_scope()
                for s in stmt["else"]:
                    self._process_stmt(s)
                self._exit_scope()

        elif kind == "while":
            cond_var = stmt.get("cond_var")
            if cond_var:
                self._use_variable(cond_var, loc)
            self._enter_scope()
            for s in stmt.get("body", []):
                self._process_stmt(s)
            self._exit_scope()

        elif kind == "match":
            scrutinee = stmt.get("scrutinee", "")
            self._use_variable(scrutinee, loc)
            arms = stmt.get("arms", [])
            self._check_match_borrows(scrutinee, arms, loc)
            for arm in arms:
                self._enter_scope()
                for s in arm.get("body", []):
                    self._process_stmt(s)
                self._exit_scope()

        elif kind == "if_let":
            scrutinee = stmt.get("scrutinee", "")
            pattern = stmt.get("pattern", {})
            self._use_variable(scrutinee, loc)
            self._enter_scope()
            self._check_if_let_borrow(scrutinee, pattern, loc)
            for s in stmt.get("body", []):
                self._process_stmt(s)
            self._exit_scope()

        elif kind == "return":
            ret_var = stmt.get("value")
            if ret_var:
                if isinstance(ret_var, str):
                    self._check_return_lifetime(ret_var, loc)
                    self._use_variable(ret_var, loc)

        elif kind == "call":
            for arg in stmt.get("args", []):
                if isinstance(arg, dict):
                    if arg.get("kind") == "move":
                        self._move_variable(arg["source"], f"_call_arg_{self._program_point}", loc)
                    elif arg.get("kind") == "borrow":
                        bk = BorrowKind.MUTABLE if arg.get("mutable") else BorrowKind.SHARED
                        self._create_borrow(
                            f"_call_arg_{self._program_point}",
                            arg["place"], bk, loc)
                elif isinstance(arg, str):
                    self._use_variable(arg, loc)

        elif kind == "field_access":
            base = stmt.get("base", "")
            self._use_variable(base, loc)

        elif kind == "reborrow":
            src_ref = stmt.get("source_ref", "")
            new_ref = stmt.get("new_ref", "")
            mutable = stmt.get("mutable", False)
            for bid, borrow in self._borrows.items():
                if borrow.borrower == src_ref and borrow.is_active:
                    new_borrow_kind = BorrowKind.MUTABLE if mutable else BorrowKind.SHARED
                    if mutable and borrow.kind != BorrowKind.MUTABLE:
                        self._result.add_violation(BorrowViolation(
                            location=loc,
                            description=f"Cannot reborrow `{src_ref}` as mutable: original borrow is shared",
                            severity=ViolationSeverity.ERROR,
                            violated_rule="E0596: cannot reborrow as mutable",
                        ))
                    else:
                        new_bid = self._create_borrow(
                            new_ref, borrow.borrowed_place, new_borrow_kind, loc)
                        if new_bid is not None:
                            self._borrows[new_bid].reborrow_of = bid
                    break

    # --- main entry ---
    def check(self, rust_ast: Any) -> BorrowCheckResult:
        self._reset()

        if isinstance(rust_ast, dict):
            stmts = rust_ast.get("body", [])
            params = rust_ast.get("params", [])
        elif isinstance(rust_ast, list):
            stmts = rust_ast
            params = []
        else:
            return self._result

        self._enter_scope()

        for param in params:
            name = param.get("name", "")
            type_name = param.get("type", "unknown")
            self._declare_variable(name, type_name, "param")

        for stmt in stmts:
            self._process_stmt(stmt)

        self._exit_scope()

        self._result.ownership_map = dict(self._ownership)
        self._result.active_borrows = [
            b for b in self._borrows.values() if b.is_active
        ]

        return self._result

    def check_function(self, func_ast: Dict[str, Any]) -> BorrowCheckResult:
        return self.check(func_ast)

    def check_program(self, program: Dict[str, Any]) -> Dict[str, BorrowCheckResult]:
        results = {}
        for func in program.get("functions", []):
            name = func.get("name", "<anonymous>")
            results[name] = self.check_function(func)
        return results

    def get_diagnostics(self) -> List[Dict[str, Any]]:
        diags = []
        for v in self._result.violations:
            diags.append({
                "level": v.severity.value,
                "message": v.description,
                "location": v.location,
                "code": v.violated_rule,
                "suggestion": v.suggestion,
            })
        return diags


# ---------------------------------------------------------------------------
# Helper: validate a simple set of Rust statements
# ---------------------------------------------------------------------------

def quick_borrow_check(stmts: List[Dict[str, Any]]) -> BorrowCheckResult:
    checker = BorrowChecker()
    return checker.check(stmts)
