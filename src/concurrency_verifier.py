"""
Concurrency verification module for C-to-Rust migration.
Verifies thread safety, mutex equivalence, atomic operations mapping,
data race detection, and concurrency primitive suggestions.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum, auto


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Severity(Enum):
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


class MemoryOrdering(Enum):
    RELAXED = "Relaxed"
    ACQUIRE = "Acquire"
    RELEASE = "Release"
    ACQ_REL = "AcqRel"
    SEQ_CST = "SeqCst"


class SuggestionCategory(Enum):
    MUTEX = auto()
    RWLOCK = auto()
    ATOMIC = auto()
    CHANNEL = auto()
    CONDVAR = auto()
    BARRIER = auto()
    ONCE = auto()
    THREAD_LOCAL = auto()
    ARC = auto()
    SCOPED_THREAD = auto()


class ChannelKind(Enum):
    BOUNDED = auto()
    UNBOUNDED = auto()
    ONESHOT = auto()
    BROADCAST = auto()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ThreadInfo:
    name: str
    line: int
    function_called: str
    args: List[str] = field(default_factory=list)
    join_line: Optional[int] = None
    detached: bool = False
    shared_variables: List[str] = field(default_factory=list)


@dataclass
class MutexInfo:
    name: str
    line: int
    lock_lines: List[int] = field(default_factory=list)
    unlock_lines: List[int] = field(default_factory=list)
    trylock_lines: List[int] = field(default_factory=list)
    is_recursive: bool = False
    guarded_variables: List[str] = field(default_factory=list)


@dataclass
class AtomicOp:
    variable: str
    operation: str
    ordering: str
    line: int
    c_expression: str
    rust_equivalent: str


@dataclass
class DataRace:
    variable: str
    write_lines: List[int] = field(default_factory=list)
    read_lines: List[int] = field(default_factory=list)
    threads_involved: List[str] = field(default_factory=list)
    severity: Severity = Severity.HIGH
    description: str = ""


@dataclass
class ThreadSafetyResult:
    is_safe: bool
    threads: List[ThreadInfo] = field(default_factory=list)
    mutexes: List[MutexInfo] = field(default_factory=list)
    data_races: List[DataRace] = field(default_factory=list)
    atomic_ops: List[AtomicOp] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    score: float = 0.0


@dataclass
class PthreadMapping:
    thread_creates: List[Dict[str, str]] = field(default_factory=list)
    thread_joins: List[Dict[str, str]] = field(default_factory=list)
    mutex_inits: List[Dict[str, str]] = field(default_factory=list)
    mutex_locks: List[Dict[str, str]] = field(default_factory=list)
    mutex_unlocks: List[Dict[str, str]] = field(default_factory=list)
    cond_inits: List[Dict[str, str]] = field(default_factory=list)
    cond_waits: List[Dict[str, str]] = field(default_factory=list)
    cond_signals: List[Dict[str, str]] = field(default_factory=list)
    barriers: List[Dict[str, str]] = field(default_factory=list)
    rwlocks: List[Dict[str, str]] = field(default_factory=list)
    rust_equivalents: Dict[str, str] = field(default_factory=dict)


@dataclass
class MutexResult:
    c_mutexes: List[MutexInfo] = field(default_factory=list)
    rust_mutexes: List[MutexInfo] = field(default_factory=list)
    matched_pairs: List[Tuple[str, str]] = field(default_factory=list)
    unmatched_c: List[str] = field(default_factory=list)
    unmatched_rust: List[str] = field(default_factory=list)
    lock_guard_usage: bool = False
    poison_handling: bool = False
    equivalence_score: float = 0.0
    issues: List[str] = field(default_factory=list)


@dataclass
class AtomicResult:
    c_atomics: List[AtomicOp] = field(default_factory=list)
    rust_atomics: List[AtomicOp] = field(default_factory=list)
    ordering_matches: List[Tuple[str, str, bool]] = field(default_factory=list)
    type_mappings: Dict[str, str] = field(default_factory=dict)
    equivalence_score: float = 0.0
    issues: List[str] = field(default_factory=list)


@dataclass
class SendSyncResult:
    types_checked: List[str] = field(default_factory=list)
    send_types: List[str] = field(default_factory=list)
    sync_types: List[str] = field(default_factory=list)
    non_send: List[str] = field(default_factory=list)
    non_sync: List[str] = field(default_factory=list)
    unsafe_impls: List[Dict[str, str]] = field(default_factory=list)
    arc_usage: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)


@dataclass
class ChannelPattern:
    kind: ChannelKind
    producer_line: int
    consumer_line: int
    c_mechanism: str
    rust_suggestion: str
    buffer_size: Optional[int] = None
    description: str = ""


@dataclass
class LockOrderViolation:
    mutex_a: str
    mutex_b: str
    location_ab: Tuple[int, int] = (0, 0)
    location_ba: Tuple[int, int] = (0, 0)
    severity: Severity = Severity.CRITICAL
    description: str = ""


@dataclass
class CondVarMapping:
    c_cond: str
    c_mutex: str
    c_wait_line: int
    c_signal_line: int
    rust_condvar: str
    rust_mutex: str
    spurious_wakeup_handled: bool = False
    predicate_checked: bool = False


@dataclass
class ConcurrencySuggestion:
    category: SuggestionCategory
    c_pattern: str
    rust_replacement: str
    rationale: str
    confidence: float = 0.0
    line: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_C_MEMORY_ORDER_MAP: Dict[str, str] = {
    "memory_order_relaxed": "Ordering::Relaxed",
    "memory_order_consume": "Ordering::Acquire",
    "memory_order_acquire": "Ordering::Acquire",
    "memory_order_release": "Ordering::Release",
    "memory_order_acq_rel": "Ordering::AcqRel",
    "memory_order_seq_cst": "Ordering::SeqCst",
    "__ATOMIC_RELAXED": "Ordering::Relaxed",
    "__ATOMIC_CONSUME": "Ordering::Acquire",
    "__ATOMIC_ACQUIRE": "Ordering::Acquire",
    "__ATOMIC_RELEASE": "Ordering::Release",
    "__ATOMIC_ACQ_REL": "Ordering::AcqRel",
    "__ATOMIC_SEQ_CST": "Ordering::SeqCst",
}

_C_ATOMIC_TYPE_MAP: Dict[str, str] = {
    "atomic_int": "AtomicI32",
    "atomic_uint": "AtomicU32",
    "atomic_long": "AtomicI64",
    "atomic_ulong": "AtomicU64",
    "atomic_short": "AtomicI16",
    "atomic_ushort": "AtomicU16",
    "atomic_char": "AtomicI8",
    "atomic_uchar": "AtomicU8",
    "atomic_bool": "AtomicBool",
    "atomic_size_t": "AtomicUsize",
    "atomic_intptr_t": "AtomicIsize",
    "atomic_uintptr_t": "AtomicUsize",
    "_Atomic int": "AtomicI32",
    "_Atomic unsigned int": "AtomicU32",
    "_Atomic long": "AtomicI64",
    "_Atomic bool": "AtomicBool",
}


def _numbered_lines(code: str) -> List[Tuple[int, str]]:
    """Return list of (1-based line number, line text)."""
    return [(i + 1, ln) for i, ln in enumerate(code.splitlines())]


def _extract_global_vars(code: str) -> Set[str]:
    """Extract likely global / file-scope variable names from C code."""
    results: Set[str] = set()
    in_function = 0
    for line in code.splitlines():
        stripped = line.strip()
        if "{" in stripped:
            in_function += stripped.count("{")
        if "}" in stripped:
            in_function -= stripped.count("}")
        if in_function <= 0:
            m = re.match(
                r'(?:static\s+|volatile\s+|extern\s+)*'
                r'(?:int|long|char|float|double|void\s*\*|unsigned\s+\w+|size_t|'
                r'atomic_\w+|_Atomic\s+\w+|pthread_\w+)\s+'
                r'(\w+)\s*[;=\[]',
                stripped,
            )
            if m:
                results.add(m.group(1))
    return results


def _extract_functions_called_in_thread(code: str, thread_func: str) -> List[str]:
    """Return variable names accessed inside a thread function body."""
    accessed: List[str] = []
    pattern = re.compile(
        rf'(?:void\s*\*\s*)?{re.escape(thread_func)}\s*\(.*?\)\s*\{{',
        re.DOTALL,
    )
    match = pattern.search(code)
    if not match:
        return accessed
    start = match.end()
    brace_depth = 1
    pos = start
    body_lines: List[str] = []
    while pos < len(code) and brace_depth > 0:
        if code[pos] == "{":
            brace_depth += 1
        elif code[pos] == "}":
            brace_depth -= 1
        pos += 1
    body = code[start:pos]
    for tok in re.findall(r'\b([a-zA-Z_]\w*)\b', body):
        if tok not in (
            "int", "long", "char", "void", "return", "if", "else",
            "while", "for", "NULL", "sizeof", "printf", "fprintf",
            "pthread_mutex_lock", "pthread_mutex_unlock",
        ) and tok not in accessed:
            accessed.append(tok)
    return accessed


def _find_guarded_vars(code: str, mutex_name: str) -> List[str]:
    """Find variables accessed between lock/unlock of a given mutex."""
    guarded: List[str] = []
    lock_pat = re.compile(
        rf'pthread_mutex_lock\s*\(\s*&?\s*{re.escape(mutex_name)}\s*\)'
    )
    unlock_pat = re.compile(
        rf'pthread_mutex_unlock\s*\(\s*&?\s*{re.escape(mutex_name)}\s*\)'
    )
    lines = code.splitlines()
    in_critical = False
    for line in lines:
        stripped = line.strip()
        if lock_pat.search(stripped):
            in_critical = True
            continue
        if unlock_pat.search(stripped):
            in_critical = False
            continue
        if in_critical:
            for var in re.findall(r'\b([a-zA-Z_]\w*)\b', stripped):
                if var not in (
                    "int", "long", "char", "void", "return", "if", "else",
                    "while", "for", "NULL", "sizeof", "printf",
                ) and var not in guarded:
                    guarded.append(var)
    return guarded


# ---------------------------------------------------------------------------
# Core verification functions
# ---------------------------------------------------------------------------

def verify_thread_safety(c_code: str, rust_code: str) -> ThreadSafetyResult:
    """Comprehensive thread safety analysis comparing C and Rust code."""
    threads = _extract_threads(c_code)
    mutexes = _extract_mutexes(c_code)
    races = detect_data_races(c_code)
    atomics = _extract_c_atomics(c_code)

    warnings: List[str] = []
    suggestions: List[str] = []

    # Check for unjoined threads
    for t in threads:
        if t.join_line is None and not t.detached:
            warnings.append(
                f"Thread '{t.name}' created at line {t.line} is never joined or detached"
            )

    # Check for unlock without lock
    for mx in mutexes:
        orphan_unlocks = [u for u in mx.unlock_lines if not any(
            l < u for l in mx.lock_lines
        )]
        if orphan_unlocks:
            warnings.append(
                f"Mutex '{mx.name}': unlock at line(s) {orphan_unlocks} "
                f"without preceding lock in linear scan"
            )

    # Check Rust side for unsafe blocks around concurrency
    unsafe_blocks = re.findall(r'unsafe\s*\{', rust_code)
    if unsafe_blocks:
        warnings.append(
            f"Rust code contains {len(unsafe_blocks)} unsafe block(s) — "
            "review for concurrency soundness"
        )

    # Check Rust uses Arc for shared ownership
    if threads and "Arc" not in rust_code and len(threads) > 1:
        suggestions.append(
            "Consider using Arc<Mutex<T>> for shared state across threads"
        )

    # Check for raw pointer sharing in Rust
    if re.search(r'\*mut\s+', rust_code) and re.search(r'thread::spawn', rust_code):
        warnings.append(
            "Raw mutable pointer used alongside thread::spawn — potential UB"
        )

    # Check Rust uses proper scoped threads or move closures
    spawns_in_rust = re.findall(r'thread::spawn\s*\(', rust_code)
    move_closures = re.findall(r'thread::spawn\s*\(\s*move\s*\|\|', rust_code)
    if len(spawns_in_rust) > len(move_closures):
        suggestions.append(
            "Some thread::spawn calls lack 'move' closures — may cause lifetime issues"
        )

    # Score calculation
    penalty = 0.0
    penalty += len(races) * 15.0
    penalty += len(warnings) * 5.0
    penalty += max(0, len(threads) - len(mutexes)) * 3.0
    score = max(0.0, min(100.0, 100.0 - penalty))

    return ThreadSafetyResult(
        is_safe=len(races) == 0 and len(warnings) == 0,
        threads=threads,
        mutexes=mutexes,
        data_races=races,
        atomic_ops=atomics,
        warnings=warnings,
        suggestions=suggestions,
        score=score,
    )


def pthread_to_rust_mapping(c_code: str) -> PthreadMapping:
    """Map pthread API calls to their Rust std equivalents."""
    mapping = PthreadMapping()

    numbered = _numbered_lines(c_code)

    # pthread_create
    for lineno, line in numbered:
        m = re.search(
            r'pthread_create\s*\(\s*&?\s*(\w+)\s*,\s*(\w+|NULL)\s*,\s*(\w+)\s*,\s*(.*?)\s*\)',
            line,
        )
        if m:
            mapping.thread_creates.append({
                "line": str(lineno),
                "thread_var": m.group(1),
                "attr": m.group(2),
                "function": m.group(3),
                "arg": m.group(4).rstrip(")"),
                "rust": f"let {m.group(1)} = thread::spawn(move || {m.group(3)}({m.group(4).rstrip(')')}));",
            })

    # pthread_join
    for lineno, line in numbered:
        m = re.search(r'pthread_join\s*\(\s*(\w+)\s*,\s*(.*?)\s*\)', line)
        if m:
            mapping.thread_joins.append({
                "line": str(lineno),
                "thread_var": m.group(1),
                "retval": m.group(2).rstrip(")"),
                "rust": f"{m.group(1)}.join().unwrap();",
            })

    # pthread_mutex_init
    for lineno, line in numbered:
        m = re.search(
            r'pthread_mutex_init\s*\(\s*&?\s*(\w+)\s*,\s*(\w+|NULL)\s*\)', line
        )
        if m:
            mapping.mutex_inits.append({
                "line": str(lineno),
                "mutex": m.group(1),
                "attr": m.group(2),
                "rust": f"let {m.group(1)} = Mutex::new(/* protected data */);",
            })

    # Static mutex initializers
    for lineno, line in numbered:
        m = re.search(
            r'pthread_mutex_t\s+(\w+)\s*=\s*PTHREAD_MUTEX_INITIALIZER', line
        )
        if m:
            mapping.mutex_inits.append({
                "line": str(lineno),
                "mutex": m.group(1),
                "attr": "default",
                "rust": f"static {m.group(1).upper()}: Mutex<()> = Mutex::new(());",
            })

    # pthread_mutex_lock
    for lineno, line in numbered:
        m = re.search(r'pthread_mutex_lock\s*\(\s*&?\s*(\w+)\s*\)', line)
        if m:
            mapping.mutex_locks.append({
                "line": str(lineno),
                "mutex": m.group(1),
                "rust": f"let _guard = {m.group(1)}.lock().unwrap();",
            })

    # pthread_mutex_unlock
    for lineno, line in numbered:
        m = re.search(r'pthread_mutex_unlock\s*\(\s*&?\s*(\w+)\s*\)', line)
        if m:
            mapping.mutex_unlocks.append({
                "line": str(lineno),
                "mutex": m.group(1),
                "rust": "// guard dropped automatically (drop(_guard);)",
            })

    # pthread_cond_init
    for lineno, line in numbered:
        m = re.search(
            r'pthread_cond_init\s*\(\s*&?\s*(\w+)\s*,\s*(\w+|NULL)\s*\)', line
        )
        if m:
            mapping.cond_inits.append({
                "line": str(lineno),
                "cond": m.group(1),
                "rust": f"let {m.group(1)} = Condvar::new();",
            })

    # pthread_cond_wait
    for lineno, line in numbered:
        m = re.search(
            r'pthread_cond_wait\s*\(\s*&?\s*(\w+)\s*,\s*&?\s*(\w+)\s*\)', line
        )
        if m:
            mapping.cond_waits.append({
                "line": str(lineno),
                "cond": m.group(1),
                "mutex": m.group(2),
                "rust": f"guard = {m.group(1)}.wait(guard).unwrap();",
            })

    # pthread_cond_signal / broadcast
    for lineno, line in numbered:
        m = re.search(r'pthread_cond_(signal|broadcast)\s*\(\s*&?\s*(\w+)\s*\)', line)
        if m:
            kind = m.group(1)
            rust_fn = "notify_one" if kind == "signal" else "notify_all"
            mapping.cond_signals.append({
                "line": str(lineno),
                "cond": m.group(2),
                "kind": kind,
                "rust": f"{m.group(2)}.{rust_fn}();",
            })

    # pthread_barrier_init
    for lineno, line in numbered:
        m = re.search(
            r'pthread_barrier_init\s*\(\s*&?\s*(\w+)\s*,\s*\w+\s*,\s*(\d+)\s*\)', line
        )
        if m:
            mapping.barriers.append({
                "line": str(lineno),
                "barrier": m.group(1),
                "count": m.group(2),
                "rust": f"let {m.group(1)} = Arc::new(Barrier::new({m.group(2)}));",
            })

    # pthread_rwlock
    for lineno, line in numbered:
        m = re.search(r'pthread_rwlock_(rd|wr)lock\s*\(\s*&?\s*(\w+)\s*\)', line)
        if m:
            kind = "read" if m.group(1) == "rd" else "write"
            mapping.rwlocks.append({
                "line": str(lineno),
                "rwlock": m.group(2),
                "kind": kind,
                "rust": f"let _guard = {m.group(2)}.{kind}().unwrap();",
            })

    # Build equivalence table
    mapping.rust_equivalents = {
        "pthread_create": "std::thread::spawn",
        "pthread_join": "JoinHandle::join",
        "pthread_mutex_init": "std::sync::Mutex::new",
        "pthread_mutex_lock": "Mutex::lock",
        "pthread_mutex_unlock": "/* MutexGuard drop */",
        "pthread_mutex_trylock": "Mutex::try_lock",
        "pthread_cond_init": "std::sync::Condvar::new",
        "pthread_cond_wait": "Condvar::wait",
        "pthread_cond_signal": "Condvar::notify_one",
        "pthread_cond_broadcast": "Condvar::notify_all",
        "pthread_rwlock_rdlock": "RwLock::read",
        "pthread_rwlock_wrlock": "RwLock::write",
        "pthread_barrier_init": "std::sync::Barrier::new",
        "pthread_barrier_wait": "Barrier::wait",
        "pthread_key_create": "thread_local! macro",
        "pthread_setspecific": "LocalKey::with",
        "pthread_getspecific": "LocalKey::with",
    }

    return mapping


def mutex_verification(c_code: str, rust_code: str) -> MutexResult:
    """Verify mutex usage equivalence between C and Rust code."""
    result = MutexResult()

    result.c_mutexes = _extract_mutexes(c_code)

    # Extract Rust mutex declarations
    rust_numbered = _numbered_lines(rust_code)
    for lineno, line in rust_numbered:
        m = re.search(r'(?:let|static)\s+(?:mut\s+)?(\w+)\s*[:=].*Mutex::new', line)
        if m:
            mx = MutexInfo(name=m.group(1), line=lineno)
            # Find lock() calls for this mutex
            for ln2, l2 in rust_numbered:
                if re.search(rf'{re.escape(m.group(1))}\.lock\(\)', l2):
                    mx.lock_lines.append(ln2)
                if re.search(rf'{re.escape(m.group(1))}\.try_lock\(\)', l2):
                    mx.trylock_lines.append(ln2)
            result.rust_mutexes.append(mx)

    # Check for lock guard usage pattern
    result.lock_guard_usage = bool(
        re.search(r'\.lock\(\)\s*\.unwrap\(\)', rust_code)
        or re.search(r'let\s+_?\w+\s*=\s*\w+\.lock\(\)', rust_code)
    )

    # Check for poison handling
    result.poison_handling = bool(
        re.search(r'\.lock\(\)\s*\.unwrap_or_else\(', rust_code)
        or re.search(r'PoisonError', rust_code)
        or re.search(r'\.is_poisoned\(\)', rust_code)
    )

    # Match C mutexes to Rust mutexes by name similarity
    c_names = {mx.name for mx in result.c_mutexes}
    r_names = {mx.name for mx in result.rust_mutexes}

    for cn in c_names:
        # Try exact match or common transformations
        candidates = [
            cn,
            cn.lower(),
            cn.replace("_mutex", ""),
            cn.replace("mtx", "mutex"),
            f"{cn}_lock",
        ]
        matched = False
        for candidate in candidates:
            if candidate in r_names:
                result.matched_pairs.append((cn, candidate))
                r_names.discard(candidate)
                matched = True
                break
        if not matched:
            result.unmatched_c.append(cn)

    result.unmatched_rust = list(r_names)

    # Verify lock/unlock pairing in C
    for mx in result.c_mutexes:
        if len(mx.lock_lines) != len(mx.unlock_lines):
            result.issues.append(
                f"C mutex '{mx.name}': {len(mx.lock_lines)} lock(s) vs "
                f"{len(mx.unlock_lines)} unlock(s) — possible leak or double-unlock"
            )

    # Verify Rust doesn't have manual unlock attempts
    if re.search(r'\.unlock\(\)', rust_code):
        result.issues.append(
            "Rust code calls .unlock() explicitly — prefer drop(guard) or scope-based release"
        )

    # Check that Rust wraps shared data inside Mutex
    if result.c_mutexes and not result.rust_mutexes:
        result.issues.append(
            "C code uses mutexes but no Mutex found in Rust — "
            "data may be unprotected"
        )

    # Score
    total = max(len(result.c_mutexes), 1)
    matched = len(result.matched_pairs)
    issue_penalty = len(result.issues) * 10.0
    result.equivalence_score = max(
        0.0, min(100.0, (matched / total) * 100.0 - issue_penalty)
    )

    return result


def atomic_verification(c_code: str, rust_code: str) -> AtomicResult:
    """Verify atomic operations mapping from C11 atomics to Rust atomics."""
    result = AtomicResult()

    result.c_atomics = _extract_c_atomics(c_code)

    # Extract Rust atomics
    rust_numbered = _numbered_lines(rust_code)
    rust_atomic_ops = [
        ("load", r'(\w+)\.load\(Ordering::(\w+)\)'),
        ("store", r'(\w+)\.store\(.*?,\s*Ordering::(\w+)\)'),
        ("fetch_add", r'(\w+)\.fetch_add\(.*?,\s*Ordering::(\w+)\)'),
        ("fetch_sub", r'(\w+)\.fetch_sub\(.*?,\s*Ordering::(\w+)\)'),
        ("fetch_and", r'(\w+)\.fetch_and\(.*?,\s*Ordering::(\w+)\)'),
        ("fetch_or", r'(\w+)\.fetch_or\(.*?,\s*Ordering::(\w+)\)'),
        ("fetch_xor", r'(\w+)\.fetch_xor\(.*?,\s*Ordering::(\w+)\)'),
        ("compare_exchange", r'(\w+)\.compare_exchange\(.*?Ordering::(\w+)'),
        ("swap", r'(\w+)\.swap\(.*?,\s*Ordering::(\w+)\)'),
    ]
    for op_name, pattern in rust_atomic_ops:
        for lineno, line in rust_numbered:
            m = re.search(pattern, line)
            if m:
                result.rust_atomics.append(AtomicOp(
                    variable=m.group(1),
                    operation=op_name,
                    ordering=m.group(2),
                    line=lineno,
                    c_expression="",
                    rust_equivalent=line.strip(),
                ))

    # Build type mapping from what's present
    for lineno, line in _numbered_lines(c_code):
        for c_type, rust_type in _C_ATOMIC_TYPE_MAP.items():
            if c_type in line:
                result.type_mappings[c_type] = rust_type

    # Check ordering equivalence
    for c_op in result.c_atomics:
        rust_ordering = _C_MEMORY_ORDER_MAP.get(c_op.ordering, "")
        matched_rust = [
            r for r in result.rust_atomics
            if r.operation == c_op.operation or (
                c_op.operation == "load" and r.operation == "load"
            )
        ]
        for r_op in matched_rust:
            expected = rust_ordering.replace("Ordering::", "")
            matches = r_op.ordering == expected
            result.ordering_matches.append((c_op.ordering, r_op.ordering, matches))
            if not matches:
                result.issues.append(
                    f"Ordering mismatch: C uses {c_op.ordering} but Rust uses "
                    f"Ordering::{r_op.ordering} at line {r_op.line}"
                )

    # Check for missing fence translations
    if re.search(r'atomic_thread_fence', c_code):
        if not re.search(r'fence\(', rust_code) and not re.search(r'compiler_fence', rust_code):
            result.issues.append(
                "C code uses atomic_thread_fence but no corresponding "
                "std::sync::atomic::fence found in Rust"
            )

    total = max(len(result.c_atomics), 1)
    mismatches = sum(1 for _, _, m in result.ordering_matches if not m)
    result.equivalence_score = max(
        0.0, 100.0 - (mismatches / total) * 50.0 - len(result.issues) * 10.0
    )

    return result


def detect_data_races(c_code: str) -> List[DataRace]:
    """Detect potential data races in C code."""
    races: List[DataRace] = []
    globals = _extract_global_vars(c_code)
    numbered = _numbered_lines(c_code)

    # Find thread functions
    thread_funcs: Dict[str, str] = {}
    for _, line in numbered:
        m = re.search(r'pthread_create\s*\(.*?,\s*\w+\s*,\s*(\w+)\s*,', line)
        if m:
            thread_funcs[m.group(1)] = m.group(1)

    if not thread_funcs:
        return races

    # Extract mutex-protected regions
    protected_vars: Set[str] = set()
    in_critical = False
    for _, line in numbered:
        if re.search(r'pthread_mutex_lock', line):
            in_critical = True
            continue
        if re.search(r'pthread_mutex_unlock', line):
            in_critical = False
            continue
        if in_critical:
            for var in globals:
                if re.search(rf'\b{re.escape(var)}\b', line):
                    protected_vars.add(var)

    # Find atomics — they are safe
    atomic_vars: Set[str] = set()
    for _, line in numbered:
        m = re.search(r'(?:atomic_\w+|_Atomic\s+\w+)\s+(\w+)', line)
        if m:
            atomic_vars.add(m.group(1))

    # Check each global for unprotected access across thread functions
    for var in globals:
        if var in protected_vars or var in atomic_vars:
            continue

        write_lines: List[int] = []
        read_lines: List[int] = []
        accessing_funcs: List[str] = []

        for func_name in thread_funcs:
            body = _get_function_body(c_code, func_name)
            if not body:
                continue
            body_numbered = _numbered_lines(body)
            for ln, bl in body_numbered:
                if re.search(rf'\b{re.escape(var)}\b', bl):
                    # Determine if write or read
                    if re.search(
                        rf'{re.escape(var)}\s*[+\-*/]?=(?!=)', bl
                    ) or re.search(
                        rf'{re.escape(var)}\s*\+\+|{re.escape(var)}\s*--|\+\+\s*{re.escape(var)}|--\s*{re.escape(var)}',
                        bl,
                    ):
                        write_lines.append(ln)
                    else:
                        read_lines.append(ln)
                    if func_name not in accessing_funcs:
                        accessing_funcs.append(func_name)

        if len(accessing_funcs) >= 2 and write_lines:
            severity = Severity.CRITICAL if len(write_lines) > 1 else Severity.HIGH
            races.append(DataRace(
                variable=var,
                write_lines=write_lines,
                read_lines=read_lines,
                threads_involved=accessing_funcs,
                severity=severity,
                description=(
                    f"Variable '{var}' accessed by threads "
                    f"{', '.join(accessing_funcs)} without synchronization"
                ),
            ))

    # Detect volatile misuse as synchronization
    for lineno, line in numbered:
        if re.search(r'volatile\s+(?:int|long|char|unsigned)', line):
            m = re.search(r'volatile\s+\w+\s+(\w+)', line)
            if m and m.group(1) not in atomic_vars:
                races.append(DataRace(
                    variable=m.group(1),
                    write_lines=[lineno],
                    read_lines=[],
                    threads_involved=["(volatile misuse)"],
                    severity=Severity.MEDIUM,
                    description=(
                        f"Variable '{m.group(1)}' uses volatile for synchronization — "
                        "volatile does not guarantee atomicity in C"
                    ),
                ))

    return races


def verify_send_sync(rust_code: str) -> SendSyncResult:
    """Check Send/Sync trait requirements in Rust code."""
    result = SendSyncResult()
    numbered = _numbered_lines(rust_code)

    # Find struct definitions
    for lineno, line in numbered:
        m = re.search(r'struct\s+(\w+)', line)
        if m:
            result.types_checked.append(m.group(1))

    # Check for explicit Send/Sync implementations
    for lineno, line in numbered:
        m = re.search(r'unsafe\s+impl\s+(Send|Sync)\s+for\s+(\w+)', line)
        if m:
            result.unsafe_impls.append({
                "trait": m.group(1),
                "type": m.group(2),
                "line": str(lineno),
            })
            result.issues.append(
                f"Unsafe impl {m.group(1)} for {m.group(2)} at line {lineno} — "
                "requires manual soundness review"
            )

    # Detect types that are likely non-Send
    non_send_patterns = [
        (r'Rc<', "Rc"),
        (r'\*mut\s+', "raw mutable pointer"),
        (r'\*const\s+', "raw const pointer"),
        (r'Cell<', "Cell"),
        (r'RefCell<', "RefCell (non-Sync)"),
        (r'MutexGuard', "MutexGuard (non-Send)"),
    ]
    for pattern, name in non_send_patterns:
        for lineno, line in numbered:
            if re.search(pattern, line):
                # Check if used across thread boundary
                if re.search(r'thread::spawn|tokio::spawn|rayon', rust_code):
                    result.non_send.append(f"{name} at line {lineno}")

    # Detect Arc usage
    for lineno, line in numbered:
        m = re.search(r'Arc::new\((.+?)\)', line)
        if m:
            result.arc_usage.append(f"Arc wrapping '{m.group(1).strip()}' at line {lineno}")

    # Check for Arc<Mutex<T>> pattern
    arc_mutex_count = len(re.findall(r'Arc<\s*Mutex<', rust_code))
    arc_rwlock_count = len(re.findall(r'Arc<\s*RwLock<', rust_code))

    # Identify Send types (types used in thread::spawn closures)
    spawn_blocks = re.findall(
        r'thread::spawn\s*\(\s*move\s*\|\|\s*\{(.*?)\}\s*\)',
        rust_code,
        re.DOTALL,
    )
    for block in spawn_blocks:
        for var in re.findall(r'\b([A-Z]\w+)\b', block):
            if var not in result.send_types and var not in ("Arc", "Mutex", "None", "Some", "Ok", "Err"):
                result.send_types.append(var)

    # Identify Sync types (types behind & references in shared contexts)
    for lineno, line in numbered:
        m = re.search(r'&\s*(Arc<[^>]+>)', line)
        if m and m.group(1) not in result.sync_types:
            result.sync_types.append(m.group(1))

    # Detect Rc used where Arc is needed
    if re.search(r'Rc<', rust_code) and re.search(r'thread::spawn', rust_code):
        result.issues.append(
            "Rc<T> is not Send — use Arc<T> for sharing across threads"
        )

    # Detect RefCell in multi-threaded context
    if re.search(r'RefCell<', rust_code) and re.search(r'thread::spawn', rust_code):
        result.issues.append(
            "RefCell<T> is not Sync — use Mutex<T> or RwLock<T> for thread-safe interior mutability"
        )

    return result


def channel_pattern_detection(c_code: str) -> List[ChannelPattern]:
    """Detect pipe/queue patterns in C code convertible to Rust channels."""
    patterns: List[ChannelPattern] = []
    numbered = _numbered_lines(c_code)

    # Detect pipe() usage
    for lineno, line in numbered:
        if re.search(r'\bpipe\s*\(', line):
            m = re.search(r'pipe\s*\(\s*(\w+)\s*\)', line)
            var = m.group(1) if m else "fd"
            # Find corresponding read/write
            read_line = 0
            write_line = 0
            for ln2, l2 in numbered:
                if re.search(rf'read\s*\(\s*{re.escape(var)}\s*\[\s*0\s*\]', l2):
                    read_line = ln2
                if re.search(rf'write\s*\(\s*{re.escape(var)}\s*\[\s*1\s*\]', l2):
                    write_line = ln2
            patterns.append(ChannelPattern(
                kind=ChannelKind.UNBOUNDED,
                producer_line=write_line or lineno,
                consumer_line=read_line or lineno,
                c_mechanism="pipe()",
                rust_suggestion="std::sync::mpsc::channel()",
                description=f"Unix pipe '{var}' can be replaced with mpsc channel",
            ))

    # Detect producer-consumer with mutex + condvar
    has_cond_signal = False
    has_cond_wait = False
    signal_line = 0
    wait_line = 0
    for lineno, line in numbered:
        if re.search(r'pthread_cond_signal', line):
            has_cond_signal = True
            signal_line = lineno
        if re.search(r'pthread_cond_wait', line):
            has_cond_wait = True
            wait_line = lineno

    if has_cond_signal and has_cond_wait:
        # Look for queue-like data structure
        has_queue = bool(re.search(
            r'(?:queue|buffer|ring|fifo|enqueue|dequeue|push|pop)\b',
            c_code,
            re.IGNORECASE,
        ))
        if has_queue:
            patterns.append(ChannelPattern(
                kind=ChannelKind.BOUNDED,
                producer_line=signal_line,
                consumer_line=wait_line,
                c_mechanism="mutex + condvar + queue",
                rust_suggestion="std::sync::mpsc::sync_channel(N)",
                description="Producer-consumer queue pattern detected — use bounded channel",
            ))
        else:
            patterns.append(ChannelPattern(
                kind=ChannelKind.UNBOUNDED,
                producer_line=signal_line,
                consumer_line=wait_line,
                c_mechanism="mutex + condvar",
                rust_suggestion="std::sync::mpsc::channel()",
                description="Signal/wait pattern detected — consider channel replacement",
            ))

    # Detect socketpair for IPC
    for lineno, line in numbered:
        if re.search(r'socketpair\s*\(', line):
            patterns.append(ChannelPattern(
                kind=ChannelKind.UNBOUNDED,
                producer_line=lineno,
                consumer_line=lineno,
                c_mechanism="socketpair()",
                rust_suggestion="std::sync::mpsc::channel() or crossbeam::channel",
                description="socketpair IPC can be replaced with in-process channel",
            ))

    # Detect message queue (mq_open)
    for lineno, line in numbered:
        if re.search(r'mq_open\s*\(', line):
            m = re.search(r'\.mq_maxmsg\s*=\s*(\d+)', c_code)
            buf_size = int(m.group(1)) if m else None
            patterns.append(ChannelPattern(
                kind=ChannelKind.BOUNDED,
                producer_line=lineno,
                consumer_line=lineno,
                c_mechanism="POSIX message queue (mq_open)",
                rust_suggestion=f"std::sync::mpsc::sync_channel({buf_size or 'N'})",
                buffer_size=buf_size,
                description="POSIX message queue can be replaced with bounded channel",
            ))

    # Detect eventfd for signaling
    for lineno, line in numbered:
        if re.search(r'eventfd\s*\(', line):
            patterns.append(ChannelPattern(
                kind=ChannelKind.ONESHOT,
                producer_line=lineno,
                consumer_line=lineno,
                c_mechanism="eventfd()",
                rust_suggestion="tokio::sync::oneshot::channel() or std::sync::mpsc::channel()",
                description="eventfd signaling pattern — oneshot or simple channel",
            ))

    return patterns


def detect_lock_order_violations(c_code: str) -> List[LockOrderViolation]:
    """Detect potential deadlocks from inconsistent lock ordering."""
    violations: List[LockOrderViolation] = []
    numbered = _numbered_lines(c_code)

    # Build a map of lock acquisition sequences per function
    function_lock_orders: Dict[str, List[Tuple[str, int]]] = {}

    current_func: Optional[str] = None
    brace_depth = 0

    for lineno, line in numbered:
        # Track function boundaries
        func_match = re.match(r'(?:\w+\s+)+(\w+)\s*\([^)]*\)\s*\{', line)
        if func_match:
            current_func = func_match.group(1)
            brace_depth = 1
            function_lock_orders.setdefault(current_func, [])
            continue

        if current_func:
            brace_depth += line.count("{") - line.count("}")
            if brace_depth <= 0:
                current_func = None
                continue

            m = re.search(r'pthread_mutex_lock\s*\(\s*&?\s*(\w+)\s*\)', line)
            if m:
                function_lock_orders[current_func].append((m.group(1), lineno))

    # Detect AB vs BA ordering across functions
    all_sequences = list(function_lock_orders.values())
    seen_pairs: Set[Tuple[str, str]] = set()

    for seq in all_sequences:
        for i in range(len(seq)):
            for j in range(i + 1, len(seq)):
                a, a_line = seq[i]
                b, b_line = seq[j]
                if a == b:
                    continue
                pair = (a, b)
                reverse_pair = (b, a)
                if reverse_pair in seen_pairs:
                    # Found AB and BA
                    violations.append(LockOrderViolation(
                        mutex_a=a,
                        mutex_b=b,
                        location_ab=(a_line, b_line),
                        severity=Severity.CRITICAL,
                        description=(
                            f"Potential deadlock: '{a}' then '{b}' at lines "
                            f"{a_line},{b_line} conflicts with reverse ordering elsewhere"
                        ),
                    ))
                seen_pairs.add(pair)

    # Detect self-deadlock (non-recursive mutex locked twice)
    for func, seq in function_lock_orders.items():
        lock_names = [s[0] for s in seq]
        for name in set(lock_names):
            occurrences = [s for s in seq if s[0] == name]
            if len(occurrences) > 1:
                # Check if there's an unlock between them
                first_lock = occurrences[0][1]
                second_lock = occurrences[1][1]
                has_unlock = False
                for lineno, line in numbered:
                    if first_lock < lineno < second_lock:
                        if re.search(
                            rf'pthread_mutex_unlock\s*\(\s*&?\s*{re.escape(name)}\s*\)',
                            line,
                        ):
                            has_unlock = True
                            break
                if not has_unlock:
                    violations.append(LockOrderViolation(
                        mutex_a=name,
                        mutex_b=name,
                        location_ab=(first_lock, second_lock),
                        severity=Severity.CRITICAL,
                        description=(
                            f"Self-deadlock: mutex '{name}' locked twice in "
                            f"'{func}' at lines {first_lock} and {second_lock} "
                            f"without intervening unlock"
                        ),
                    ))

    return violations


def map_memory_ordering(c_code: str) -> Dict[str, str]:
    """Map C memory orderings found in code to Rust equivalents."""
    result: Dict[str, str] = {}

    for c_order, rust_order in _C_MEMORY_ORDER_MAP.items():
        if c_order in c_code:
            result[c_order] = rust_order

    # Handle implicit seq_cst (C11 atomic ops without explicit ordering)
    implicit_patterns = [
        r'atomic_load\s*\(\s*&',
        r'atomic_store\s*\(\s*&',
        r'atomic_fetch_add\s*\(\s*&',
        r'atomic_fetch_sub\s*\(\s*&',
        r'atomic_exchange\s*\(\s*&',
        r'atomic_compare_exchange_strong\s*\(\s*&',
    ]
    for pat in implicit_patterns:
        if re.search(pat, c_code):
            result["(implicit seq_cst)"] = "Ordering::SeqCst"
            break

    # Handle GCC builtins
    gcc_builtins = {
        "__sync_fetch_and_add": "Ordering::SeqCst (GCC builtin)",
        "__sync_fetch_and_sub": "Ordering::SeqCst (GCC builtin)",
        "__sync_val_compare_and_swap": "Ordering::SeqCst (GCC builtin)",
        "__sync_lock_test_and_set": "Ordering::Acquire (GCC builtin)",
        "__sync_lock_release": "Ordering::Release (GCC builtin)",
        "__sync_synchronize": "std::sync::atomic::fence(Ordering::SeqCst)",
    }
    for builtin, rust_eq in gcc_builtins.items():
        if builtin in c_code:
            result[builtin] = rust_eq

    return result


def suggest_concurrency_primitives(c_code: str) -> List[ConcurrencySuggestion]:
    """Suggest appropriate Rust concurrency primitives for C patterns."""
    suggestions: List[ConcurrencySuggestion] = []
    numbered = _numbered_lines(c_code)

    # pthread_mutex -> Mutex or RwLock
    for lineno, line in numbered:
        if re.search(r'pthread_mutex_t\s+(\w+)', line):
            m = re.search(r'pthread_mutex_t\s+(\w+)', line)
            name = m.group(1) if m else "mutex"
            # Check if mostly read access
            lock_count = len(re.findall(rf'pthread_mutex_lock.*{re.escape(name)}', c_code))
            rdlock = len(re.findall(rf'pthread_rwlock_rdlock', c_code))
            if rdlock > 0:
                suggestions.append(ConcurrencySuggestion(
                    category=SuggestionCategory.RWLOCK,
                    c_pattern=f"pthread_mutex_t {name}",
                    rust_replacement=f"std::sync::RwLock<T>",
                    rationale="Read-write lock pattern detected; RwLock allows concurrent readers",
                    confidence=0.85,
                    line=lineno,
                ))
            else:
                suggestions.append(ConcurrencySuggestion(
                    category=SuggestionCategory.MUTEX,
                    c_pattern=f"pthread_mutex_t {name}",
                    rust_replacement=f"std::sync::Mutex<T>",
                    rationale="Standard mutex — Rust Mutex wraps protected data directly",
                    confidence=0.95,
                    line=lineno,
                ))

    # pthread_rwlock -> RwLock
    for lineno, line in numbered:
        m = re.search(r'pthread_rwlock_t\s+(\w+)', line)
        if m:
            suggestions.append(ConcurrencySuggestion(
                category=SuggestionCategory.RWLOCK,
                c_pattern=f"pthread_rwlock_t {m.group(1)}",
                rust_replacement="std::sync::RwLock<T>",
                rationale="Direct mapping from POSIX rwlock to Rust RwLock",
                confidence=0.95,
                line=lineno,
            ))

    # atomic variables -> Atomic types
    for lineno, line in numbered:
        m = re.search(r'(?:atomic_(\w+)|_Atomic\s+(\w+))\s+(\w+)', line)
        if m:
            c_type = m.group(1) or m.group(2)
            var_name = m.group(3)
            rust_type = _C_ATOMIC_TYPE_MAP.get(f"atomic_{c_type}", f"Atomic{c_type.title()}")
            suggestions.append(ConcurrencySuggestion(
                category=SuggestionCategory.ATOMIC,
                c_pattern=line.strip(),
                rust_replacement=f"std::sync::atomic::{rust_type}",
                rationale=f"C11 atomic maps to Rust atomic type {rust_type}",
                confidence=0.90,
                line=lineno,
            ))

    # pthread_cond -> Condvar
    for lineno, line in numbered:
        m = re.search(r'pthread_cond_t\s+(\w+)', line)
        if m:
            suggestions.append(ConcurrencySuggestion(
                category=SuggestionCategory.CONDVAR,
                c_pattern=f"pthread_cond_t {m.group(1)}",
                rust_replacement="std::sync::Condvar",
                rationale="Condition variable maps to Condvar; pair with Mutex",
                confidence=0.90,
                line=lineno,
            ))

    # pthread_barrier -> Barrier
    for lineno, line in numbered:
        m = re.search(r'pthread_barrier_t\s+(\w+)', line)
        if m:
            suggestions.append(ConcurrencySuggestion(
                category=SuggestionCategory.BARRIER,
                c_pattern=f"pthread_barrier_t {m.group(1)}",
                rust_replacement="std::sync::Barrier",
                rationale="Direct mapping from POSIX barrier to Rust Barrier",
                confidence=0.95,
                line=lineno,
            ))

    # pthread_once -> Once
    for lineno, line in numbered:
        if re.search(r'pthread_once\b', line):
            suggestions.append(ConcurrencySuggestion(
                category=SuggestionCategory.ONCE,
                c_pattern="pthread_once()",
                rust_replacement="std::sync::Once or std::sync::OnceLock",
                rationale="One-time initialization maps to Once/OnceLock",
                confidence=0.95,
                line=lineno,
            ))

    # pthread_key_create -> thread_local!
    for lineno, line in numbered:
        if re.search(r'pthread_key_create', line):
            suggestions.append(ConcurrencySuggestion(
                category=SuggestionCategory.THREAD_LOCAL,
                c_pattern="pthread_key_create()",
                rust_replacement="thread_local! { static VAR: RefCell<T> = ... }",
                rationale="Thread-local storage maps to thread_local! macro",
                confidence=0.90,
                line=lineno,
            ))

    # Shared global pointers across threads -> Arc
    globals = _extract_global_vars(c_code)
    thread_funcs: List[str] = []
    for _, line in numbered:
        m = re.search(r'pthread_create\s*\(.*?,\s*\w+\s*,\s*(\w+)', line)
        if m:
            thread_funcs.append(m.group(1))

    for var in globals:
        accessing_threads = 0
        for func in thread_funcs:
            body = _get_function_body(c_code, func)
            if body and re.search(rf'\b{re.escape(var)}\b', body):
                accessing_threads += 1
        if accessing_threads >= 2:
            suggestions.append(ConcurrencySuggestion(
                category=SuggestionCategory.ARC,
                c_pattern=f"shared global '{var}' across {accessing_threads} threads",
                rust_replacement=f"Arc<Mutex<T>> for '{var}'",
                rationale="Shared mutable state across threads requires Arc<Mutex<T>>",
                confidence=0.85,
                line=0,
            ))

    # pipe/socketpair -> channel
    for lineno, line in numbered:
        if re.search(r'\bpipe\s*\(', line) or re.search(r'socketpair\s*\(', line):
            suggestions.append(ConcurrencySuggestion(
                category=SuggestionCategory.CHANNEL,
                c_pattern=line.strip(),
                rust_replacement="std::sync::mpsc::channel()",
                rationale="IPC pipe/socket can be replaced with in-process channel",
                confidence=0.80,
                line=lineno,
            ))

    # Scoped threads when threads access stack data
    for lineno, line in numbered:
        m = re.search(r'pthread_create\s*\(.*?,\s*\w+\s*,\s*\w+\s*,\s*&(\w+)\s*\)', line)
        if m:
            suggestions.append(ConcurrencySuggestion(
                category=SuggestionCategory.SCOPED_THREAD,
                c_pattern=f"pthread_create with stack reference &{m.group(1)}",
                rust_replacement="std::thread::scope(|s| s.spawn(|| ...))",
                rationale="Stack-borrowed data in threads maps to scoped threads",
                confidence=0.75,
                line=lineno,
            ))

    return suggestions


# ---------------------------------------------------------------------------
# Internal extraction helpers
# ---------------------------------------------------------------------------

def _extract_threads(c_code: str) -> List[ThreadInfo]:
    """Extract thread creation/join information from C code."""
    threads: List[ThreadInfo] = []
    numbered = _numbered_lines(c_code)

    for lineno, line in numbered:
        m = re.search(
            r'pthread_create\s*\(\s*&?\s*(\w+)\s*,\s*(\w+|NULL)\s*,\s*(\w+)\s*,\s*(.*?)\s*\)',
            line,
        )
        if m:
            thread_var = m.group(1)
            func = m.group(3)
            arg = m.group(4).rstrip(")")

            shared = _extract_functions_called_in_thread(c_code, func)
            globals_set = _extract_global_vars(c_code)
            shared_globals = [v for v in shared if v in globals_set]

            ti = ThreadInfo(
                name=thread_var,
                line=lineno,
                function_called=func,
                args=[arg] if arg and arg != "NULL" else [],
                shared_variables=shared_globals,
            )

            # Find join
            for ln2, l2 in numbered:
                if re.search(rf'pthread_join\s*\(\s*{re.escape(thread_var)}\b', l2):
                    ti.join_line = ln2
                    break

            # Check for detach
            for ln2, l2 in numbered:
                if re.search(rf'pthread_detach\s*\(\s*{re.escape(thread_var)}\b', l2):
                    ti.detached = True
                    break

            threads.append(ti)

    return threads


def _extract_mutexes(c_code: str) -> List[MutexInfo]:
    """Extract mutex declarations and usage from C code."""
    mutexes: List[MutexInfo] = []
    numbered = _numbered_lines(c_code)
    known_names: Set[str] = set()

    # Find declarations
    for lineno, line in numbered:
        m = re.search(r'pthread_mutex_t\s+(\w+)', line)
        if m:
            known_names.add(m.group(1))
            mutexes.append(MutexInfo(name=m.group(1), line=lineno))

    # Find init calls for undeclared mutexes
    for lineno, line in numbered:
        m = re.search(r'pthread_mutex_init\s*\(\s*&?\s*(\w+)', line)
        if m and m.group(1) not in known_names:
            known_names.add(m.group(1))
            mutexes.append(MutexInfo(name=m.group(1), line=lineno))

    # Populate lock/unlock/trylock lines
    for mx in mutexes:
        for lineno, line in numbered:
            if re.search(rf'pthread_mutex_lock\s*\(\s*&?\s*{re.escape(mx.name)}\s*\)', line):
                mx.lock_lines.append(lineno)
            if re.search(rf'pthread_mutex_unlock\s*\(\s*&?\s*{re.escape(mx.name)}\s*\)', line):
                mx.unlock_lines.append(lineno)
            if re.search(rf'pthread_mutex_trylock\s*\(\s*&?\s*{re.escape(mx.name)}\s*\)', line):
                mx.trylock_lines.append(lineno)

        # Check recursive attribute
        attr_pat = re.compile(
            rf'pthread_mutexattr_settype\s*\(.*?PTHREAD_MUTEX_RECURSIVE',
            re.DOTALL,
        )
        if attr_pat.search(c_code):
            mx.is_recursive = True

        mx.guarded_variables = _find_guarded_vars(c_code, mx.name)

    return mutexes


def _extract_c_atomics(c_code: str) -> List[AtomicOp]:
    """Extract C11 and GCC atomic operations from C code."""
    ops: List[AtomicOp] = []
    numbered = _numbered_lines(c_code)

    # C11 explicit ordering atomic operations
    c11_patterns = [
        (r'atomic_load_explicit\s*\(\s*&?\s*(\w+)\s*,\s*(\w+)\s*\)', "load"),
        (r'atomic_store_explicit\s*\(\s*&?\s*(\w+)\s*,\s*.*?,\s*(\w+)\s*\)', "store"),
        (r'atomic_fetch_add_explicit\s*\(\s*&?\s*(\w+)\s*,\s*.*?,\s*(\w+)\s*\)', "fetch_add"),
        (r'atomic_fetch_sub_explicit\s*\(\s*&?\s*(\w+)\s*,\s*.*?,\s*(\w+)\s*\)', "fetch_sub"),
        (r'atomic_fetch_and_explicit\s*\(\s*&?\s*(\w+)\s*,\s*.*?,\s*(\w+)\s*\)', "fetch_and"),
        (r'atomic_fetch_or_explicit\s*\(\s*&?\s*(\w+)\s*,\s*.*?,\s*(\w+)\s*\)', "fetch_or"),
        (r'atomic_fetch_xor_explicit\s*\(\s*&?\s*(\w+)\s*,\s*.*?,\s*(\w+)\s*\)', "fetch_xor"),
        (r'atomic_exchange_explicit\s*\(\s*&?\s*(\w+)\s*,\s*.*?,\s*(\w+)\s*\)', "swap"),
        (r'atomic_compare_exchange_strong_explicit\s*\(\s*&?\s*(\w+)\s*,.*?,\s*(\w+)\s*,\s*\w+\s*\)', "compare_exchange"),
    ]
    for pattern, op_name in c11_patterns:
        for lineno, line in numbered:
            m = re.search(pattern, line)
            if m:
                var = m.group(1)
                ordering = m.group(2)
                rust_ord = _C_MEMORY_ORDER_MAP.get(ordering, "Ordering::SeqCst")
                rust_method = op_name
                ops.append(AtomicOp(
                    variable=var,
                    operation=op_name,
                    ordering=ordering,
                    line=lineno,
                    c_expression=line.strip(),
                    rust_equivalent=f"{var}.{rust_method}({rust_ord})",
                ))

    # C11 implicit seq_cst atomic operations
    implicit_patterns = [
        (r'atomic_load\s*\(\s*&?\s*(\w+)\s*\)', "load"),
        (r'atomic_store\s*\(\s*&?\s*(\w+)\s*,', "store"),
        (r'atomic_fetch_add\s*\(\s*&?\s*(\w+)\s*,', "fetch_add"),
        (r'atomic_fetch_sub\s*\(\s*&?\s*(\w+)\s*,', "fetch_sub"),
        (r'atomic_exchange\s*\(\s*&?\s*(\w+)\s*,', "swap"),
        (r'atomic_compare_exchange_strong\s*\(\s*&?\s*(\w+)', "compare_exchange"),
    ]
    for pattern, op_name in implicit_patterns:
        for lineno, line in numbered:
            m = re.search(pattern, line)
            if m:
                var = m.group(1)
                ops.append(AtomicOp(
                    variable=var,
                    operation=op_name,
                    ordering="memory_order_seq_cst",
                    line=lineno,
                    c_expression=line.strip(),
                    rust_equivalent=f"{var}.{op_name}(Ordering::SeqCst)",
                ))

    # GCC __sync builtins
    gcc_patterns = [
        (r'__sync_fetch_and_add\s*\(\s*&?\s*(\w+)', "fetch_add"),
        (r'__sync_fetch_and_sub\s*\(\s*&?\s*(\w+)', "fetch_sub"),
        (r'__sync_fetch_and_and\s*\(\s*&?\s*(\w+)', "fetch_and"),
        (r'__sync_fetch_and_or\s*\(\s*&?\s*(\w+)', "fetch_or"),
        (r'__sync_fetch_and_xor\s*\(\s*&?\s*(\w+)', "fetch_xor"),
        (r'__sync_val_compare_and_swap\s*\(\s*&?\s*(\w+)', "compare_exchange"),
        (r'__sync_lock_test_and_set\s*\(\s*&?\s*(\w+)', "swap"),
    ]
    for pattern, op_name in gcc_patterns:
        for lineno, line in numbered:
            m = re.search(pattern, line)
            if m:
                ops.append(AtomicOp(
                    variable=m.group(1),
                    operation=op_name,
                    ordering="__ATOMIC_SEQ_CST",
                    line=lineno,
                    c_expression=line.strip(),
                    rust_equivalent=f"{m.group(1)}.{op_name}(Ordering::SeqCst)",
                ))

    # GCC __atomic builtins with explicit ordering
    gcc_atomic_patterns = [
        (r'__atomic_load_n\s*\(\s*&?\s*(\w+)\s*,\s*(\w+)\s*\)', "load"),
        (r'__atomic_store_n\s*\(\s*&?\s*(\w+)\s*,\s*.*?,\s*(\w+)\s*\)', "store"),
        (r'__atomic_fetch_add\s*\(\s*&?\s*(\w+)\s*,\s*.*?,\s*(\w+)\s*\)', "fetch_add"),
        (r'__atomic_fetch_sub\s*\(\s*&?\s*(\w+)\s*,\s*.*?,\s*(\w+)\s*\)', "fetch_sub"),
        (r'__atomic_exchange_n\s*\(\s*&?\s*(\w+)\s*,\s*.*?,\s*(\w+)\s*\)', "swap"),
    ]
    for pattern, op_name in gcc_atomic_patterns:
        for lineno, line in numbered:
            m = re.search(pattern, line)
            if m:
                var = m.group(1)
                ordering = m.group(2)
                rust_ord = _C_MEMORY_ORDER_MAP.get(ordering, "Ordering::SeqCst")
                ops.append(AtomicOp(
                    variable=var,
                    operation=op_name,
                    ordering=ordering,
                    line=lineno,
                    c_expression=line.strip(),
                    rust_equivalent=f"{var}.{op_name}({rust_ord})",
                ))

    return ops


def _get_function_body(code: str, func_name: str) -> Optional[str]:
    """Extract the body of a function by name."""
    pattern = re.compile(
        rf'(?:\w[\w\s\*]*)\s+{re.escape(func_name)}\s*\([^)]*\)\s*\{{',
        re.DOTALL,
    )
    match = pattern.search(code)
    if not match:
        return None
    start = match.end()
    brace_depth = 1
    pos = start
    while pos < len(code) and brace_depth > 0:
        if code[pos] == "{":
            brace_depth += 1
        elif code[pos] == "}":
            brace_depth -= 1
        pos += 1
    return code[start:pos - 1]
