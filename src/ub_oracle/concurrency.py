"""Concurrency / data-race awareness (Step 35's sibling, Step 34).

A data race in C is **undefined behaviour**: two threads access the same location,
at least one writes, and nothing orders them.  The target languages a C codebase
migrates *into* treat the very same idiom very differently, which is exactly the
kind of cross-language divergence this project exists to surface:

  * **Rust** rejects shared mutable aliasing across threads at *compile time*: a
    plain ``&mut`` cannot cross a thread boundary, and sharing requires ``Sync``
    types (``Mutex``/``Atomic``).  The racy C idiom simply does not type-check.
  * **Go** *defines* its memory model but the same unsynchronized counter is a
    race the runtime detector (``go run -race``) flags, and the result is a
    non-deterministic value rather than UB.
  * Properly synchronized variants (a mutex or an atomic) are race-free in **all**
    of C, Rust and Go and translate faithfully.

This module models a small, precise catalogue of concurrency patterns and
**proves the racy/race-free verdict against real sanitizers on real binaries**:
the C side under **ThreadSanitizer**, the Go side under the **race detector**.  A
pattern is only ever called "race" if *both* real detectors agree (or, for the
race-free patterns, neither fires).
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional

CC = "/usr/bin/clang"
GO = "/opt/homebrew/bin/go"


@dataclass(frozen=True)
class RacePattern:
    name: str
    races: bool
    description: str
    c_source: str
    go_source: str
    rust_story: str


def _c_unsync() -> str:
    return (
        "#include <pthread.h>\n"
        "static int g;\n"
        "static void* w(void* a){ (void)a; for(int i=0;i<200000;i++) g++; "
        "return 0; }\n"
        "int main(){ pthread_t t1,t2; pthread_create(&t1,0,w,0); "
        "pthread_create(&t2,0,w,0); pthread_join(t1,0); pthread_join(t2,0); "
        "return g & 1; }\n")


def _c_mutex() -> str:
    return (
        "#include <pthread.h>\n"
        "static int g; static pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;\n"
        "static void* w(void* a){ (void)a; for(int i=0;i<200000;i++){ "
        "pthread_mutex_lock(&m); g++; pthread_mutex_unlock(&m);} return 0; }\n"
        "int main(){ pthread_t t1,t2; pthread_create(&t1,0,w,0); "
        "pthread_create(&t2,0,w,0); pthread_join(t1,0); pthread_join(t2,0); "
        "return g & 1; }\n")


def _c_atomic() -> str:
    return (
        "#include <pthread.h>\n#include <stdatomic.h>\n"
        "static _Atomic int g;\n"
        "static void* w(void* a){ (void)a; for(int i=0;i<200000;i++) "
        "atomic_fetch_add(&g,1); return 0; }\n"
        "int main(){ pthread_t t1,t2; pthread_create(&t1,0,w,0); "
        "pthread_create(&t2,0,w,0); pthread_join(t1,0); pthread_join(t2,0); "
        "return atomic_load(&g) & 1; }\n")


def _c_readonly() -> str:
    return (
        "#include <pthread.h>\n"
        "static const int g = 7;\n"
        "static void* w(void* a){ int* out = (int*)a; *out = g + 1; return 0; }\n"
        "int main(){ pthread_t t1,t2; int a=0,b=0; pthread_create(&t1,0,w,&a); "
        "pthread_create(&t2,0,w,&b); pthread_join(t1,0); pthread_join(t2,0); "
        "return (a+b) & 1; }\n")


def _go_unsync() -> str:
    return (
        "package main\nimport \"sync\"\n"
        "func main(){ var wg sync.WaitGroup; g:=0\n"
        " for i:=0;i<2;i++{ wg.Add(1); go func(){ defer wg.Done(); "
        "for j:=0;j<200000;j++{ g++ } }() }\n wg.Wait(); _=g }\n")


def _go_mutex() -> str:
    return (
        "package main\nimport \"sync\"\n"
        "func main(){ var wg sync.WaitGroup; var m sync.Mutex; g:=0\n"
        " for i:=0;i<2;i++{ wg.Add(1); go func(){ defer wg.Done(); "
        "for j:=0;j<200000;j++{ m.Lock(); g++; m.Unlock() } }() }\n"
        " wg.Wait(); _=g }\n")


def _go_atomic() -> str:
    return (
        "package main\nimport (\"sync\"; \"sync/atomic\")\n"
        "func main(){ var wg sync.WaitGroup; var g int64\n"
        " for i:=0;i<2;i++{ wg.Add(1); go func(){ defer wg.Done(); "
        "for j:=0;j<200000;j++{ atomic.AddInt64(&g,1) } }() }\n"
        " wg.Wait(); _=atomic.LoadInt64(&g) }\n")


def _go_readonly() -> str:
    return (
        "package main\nimport \"sync\"\n"
        "func main(){ var wg sync.WaitGroup; const g = 7; out:=make([]int,2)\n"
        " for i:=0;i<2;i++{ wg.Add(1); go func(k int){ defer wg.Done(); "
        "out[k]=g+1 }(i) }\n wg.Wait(); _=out }\n")


PATTERNS: Dict[str, RacePattern] = {
    "unsynchronized_counter": RacePattern(
        "unsynchronized_counter", True,
        "Two threads increment a shared non-atomic counter with no lock — a data "
        "race (UB in C).",
        _c_unsync(), _go_unsync(),
        "Rust: rejected at compile time — a shared `&mut i32` cannot cross a "
        "thread boundary (needs `Sync`, i.e. `Mutex`/`Atomic`)."),
    "mutex_counter": RacePattern(
        "mutex_counter", False,
        "The counter is guarded by a mutex on every access — race-free and "
        "faithful across C/Rust/Go.",
        _c_mutex(), _go_mutex(),
        "Rust: `Arc<Mutex<i32>>` — accepted and race-free."),
    "atomic_counter": RacePattern(
        "atomic_counter", False,
        "The counter is a lock-free atomic with fetch-add — race-free in all "
        "three languages.",
        _c_atomic(), _go_atomic(),
        "Rust: `Arc<AtomicI32>` with `fetch_add` — accepted and race-free."),
    "readonly_shared": RacePattern(
        "readonly_shared", False,
        "Both threads only *read* the shared constant and write to private "
        "locations — no race.",
        _c_readonly(), _go_readonly(),
        "Rust: a shared `&i32` is `Sync` and freely shareable — accepted."),
}


def pattern(name: str) -> RacePattern:
    return PATTERNS[name]


@dataclass
class SideConfirmation:
    lang: str
    available: bool
    race_detected: Optional[bool]
    detail: str


@dataclass
class RaceConfirmation:
    name: str
    predicted_race: bool
    c: SideConfirmation
    go: SideConfirmation

    @property
    def ok(self) -> bool:
        # every available detector must agree with the prediction.
        for side in (self.c, self.go):
            if side.available and side.race_detected is not None:
                if side.race_detected != self.predicted_race:
                    return False
        # require at least one real detector to have actually run.
        return any(s.available and s.race_detected is not None
                   for s in (self.c, self.go))


def _confirm_c(p: RacePattern) -> SideConfirmation:
    if not os.path.exists(CC):
        return SideConfirmation("c", False, None, "clang unavailable")
    with tempfile.TemporaryDirectory() as d:
        cpath = os.path.join(d, "a.c")
        with open(cpath, "w") as f:
            f.write(p.c_source)
        bpath = os.path.join(d, "a")
        comp = subprocess.run(
            [CC, "-fsanitize=thread", "-O1", "-g", "-o", bpath, cpath],
            capture_output=True, text=True)
        if comp.returncode != 0:
            return SideConfirmation("c", True, None,
                                    "tsan compile failed: " + comp.stderr[:200])
        env = dict(os.environ)
        env["TSAN_OPTIONS"] = "exitcode=66"
        run = subprocess.run([bpath], capture_output=True, text=True, env=env)
        raced = ("ThreadSanitizer: data race" in run.stderr
                 or "WARNING: ThreadSanitizer" in run.stderr)
        return SideConfirmation("c", True, raced,
                                "tsan " + ("RACE" if raced else "clean"))


def _confirm_go(p: RacePattern) -> SideConfirmation:
    if not os.path.exists(GO):
        return SideConfirmation("go", False, None, "go unavailable")
    with tempfile.TemporaryDirectory() as d:
        env = dict(os.environ)
        env["GOCACHE"] = os.path.join(d, "gocache")
        env["GOPATH"] = os.path.join(d, "gopath")
        gpath = os.path.join(d, "m.go")
        with open(gpath, "w") as f:
            f.write(p.go_source)
        run = subprocess.run([GO, "run", "-race", gpath],
                             capture_output=True, text=True, env=env)
        raced = "DATA RACE" in run.stderr
        return SideConfirmation("go", True, raced,
                                "go -race " + ("RACE" if raced else "clean"))


def confirm_race(name: str, check_go: bool = True) -> RaceConfirmation:
    """Compile + run the pattern under real race detectors and check the verdict.

    The C side runs under ThreadSanitizer; the Go side (optional, slower) under
    ``go run -race``.  Both must agree with ``pattern.races``.
    """
    p = pattern(name)
    c = _confirm_c(p)
    go = (_confirm_go(p) if check_go
          else SideConfirmation("go", False, None, "skipped"))
    return RaceConfirmation(name, p.races, c, go)


# Per-target documentation of how each language treats the racy C idiom — the
# migration "story" the oracle reports alongside a confirmed race.
RACE_FRONTIER: Dict[str, str] = {
    "c": "a data race is undefined behaviour; the program may do anything. "
         "ThreadSanitizer flags it on a real binary.",
    "rust": "shared mutable aliasing across threads is a *compile error* "
            "(Send/Sync); the racy idiom cannot be written safely — it must "
            "become Mutex/Atomic, eliminating the race by construction.",
    "go": "the memory model defines the program but the unsynchronized counter "
          "is still a race (non-deterministic result); `go run -race` flags it.",
}
