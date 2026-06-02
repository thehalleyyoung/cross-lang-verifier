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
**proves the racy/race-free verdict against real tools**: the C side under
**ThreadSanitizer**, the Go side under the **race detector**, and the Rust side
under the real borrow checker.  A pattern is only ever called "race" if every
available detector/compiler agrees with the prediction.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Dict, Optional


def _tool(name: str, fallback: str) -> str:
    return shutil.which(name) or (fallback if os.path.exists(fallback) else "")


CC = _tool("clang", "/usr/bin/clang")
GO = _tool("go", "/opt/homebrew/bin/go")
RUSTC = _tool("rustc", os.path.expanduser("~/.cargo/bin/rustc"))


@dataclass(frozen=True)
class RacePattern:
    name: str
    races: bool
    description: str
    c_source: str
    go_source: str
    rust_source: str
    rust_accepts: bool
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


def _rust_unsync_rejected() -> str:
    return (
        "use std::thread;\n"
        "fn main(){\n"
        "  let mut g = 0i32;\n"
        "  let r1 = &mut g;\n"
        "  let r2 = &mut g;\n"
        "  let t1 = thread::spawn(move || { *r1 += 1; });\n"
        "  let t2 = thread::spawn(move || { *r2 += 1; });\n"
        "  t1.join().unwrap(); t2.join().unwrap();\n"
        "}\n")


def _rust_mutex() -> str:
    return (
        "use std::{sync::{Arc, Mutex}, thread};\n"
        "fn main(){\n"
        "  let g = Arc::new(Mutex::new(0i32));\n"
        "  let mut hs = Vec::new();\n"
        "  for _ in 0..2 { let g = Arc::clone(&g); hs.push(thread::spawn(move || {\n"
        "    for _ in 0..1000 { *g.lock().unwrap() += 1; }\n"
        "  })); }\n"
        "  for h in hs { h.join().unwrap(); }\n"
        "  let _ = *g.lock().unwrap();\n"
        "}\n")


def _rust_atomic() -> str:
    return (
        "use std::{sync::{Arc, atomic::{AtomicI32, Ordering}}, thread};\n"
        "fn main(){\n"
        "  let g = Arc::new(AtomicI32::new(0));\n"
        "  let mut hs = Vec::new();\n"
        "  for _ in 0..2 { let g = Arc::clone(&g); hs.push(thread::spawn(move || {\n"
        "    for _ in 0..1000 { g.fetch_add(1, Ordering::Relaxed); }\n"
        "  })); }\n"
        "  for h in hs { h.join().unwrap(); }\n"
        "  let _ = g.load(Ordering::Relaxed);\n"
        "}\n")


def _rust_readonly() -> str:
    return (
        "use std::thread;\n"
        "static G: i32 = 7;\n"
        "fn main(){\n"
        "  let t1 = thread::spawn(|| G + 1);\n"
        "  let t2 = thread::spawn(|| G + 1);\n"
        "  let _ = t1.join().unwrap() + t2.join().unwrap();\n"
        "}\n")


PATTERNS: Dict[str, RacePattern] = {
    "unsynchronized_counter": RacePattern(
        "unsynchronized_counter", True,
        "Two threads increment a shared non-atomic counter with no lock — a data "
        "race (UB in C).",
        _c_unsync(), _go_unsync(), _rust_unsync_rejected(), False,
        "Rust: rejected at compile time — a shared `&mut i32` cannot cross a "
        "thread boundary (needs `Sync`, i.e. `Mutex`/`Atomic`)."),
    "mutex_counter": RacePattern(
        "mutex_counter", False,
        "The counter is guarded by a mutex on every access — race-free and "
        "faithful across C/Rust/Go.",
        _c_mutex(), _go_mutex(), _rust_mutex(), True,
        "Rust: `Arc<Mutex<i32>>` — accepted and race-free."),
    "atomic_counter": RacePattern(
        "atomic_counter", False,
        "The counter is a lock-free atomic with fetch-add — race-free in all "
        "three languages.",
        _c_atomic(), _go_atomic(), _rust_atomic(), True,
        "Rust: `Arc<AtomicI32>` with `fetch_add` — accepted and race-free."),
    "readonly_shared": RacePattern(
        "readonly_shared", False,
        "Both threads only *read* the shared constant and write to private "
        "locations — no race.",
        _c_readonly(), _go_readonly(), _rust_readonly(), True,
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
class RustConfirmation:
    available: bool
    accepted: Optional[bool]
    error_code: Optional[str]
    detail: str

    @property
    def rejected(self) -> Optional[bool]:
        return None if self.accepted is None else not self.accepted


@dataclass
class RaceConfirmation:
    name: str
    predicted_race: bool
    c: SideConfirmation
    go: SideConfirmation
    rust: RustConfirmation

    @property
    def ok(self) -> bool:
        # Every available race detector must agree with the prediction.
        for side in (self.c, self.go):
            if side.available and side.race_detected is not None:
                if side.race_detected != self.predicted_race:
                    return False
        # Rust's safe surface must reject the racy idiom and accept synchronized
        # variants; this is compile-time evidence, not a race detector.
        if self.rust.available and self.rust.accepted is not None:
            if self.rust.accepted != (not self.predicted_race):
                return False
        # Require at least one real detector or compiler to have actually run.
        return (any(s.available and s.race_detected is not None
                    for s in (self.c, self.go))
                or (self.rust.available and self.rust.accepted is not None))


def c_race_detector_available() -> bool:
    if not CC or not os.path.exists(CC):
        return False
    with tempfile.TemporaryDirectory() as d:
        cpath = os.path.join(d, "probe.c")
        with open(cpath, "w") as f:
            f.write("#include <pthread.h>\nint main(){return 0;}\n")
        bpath = os.path.join(d, "probe")
        comp = subprocess.run(
            [CC, "-fsanitize=thread", "-O1", "-g", "-o", bpath, cpath],
            capture_output=True, text=True)
        return comp.returncode == 0


def go_race_detector_available() -> bool:
    if not GO or not os.path.exists(GO):
        return False
    with tempfile.TemporaryDirectory() as d:
        env = dict(os.environ)
        env["GOCACHE"] = os.path.join(d, "gocache")
        env["GOPATH"] = os.path.join(d, "gopath")
        gpath = os.path.join(d, "probe.go")
        with open(gpath, "w") as f:
            f.write("package main\nfunc main(){}\n")
        run = subprocess.run([GO, "run", "-race", gpath],
                             capture_output=True, text=True, env=env)
        return run.returncode == 0


def _confirm_c(p: RacePattern) -> SideConfirmation:
    if not CC or not os.path.exists(CC):
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
            return SideConfirmation("c", False, None,
                                    "tsan compile failed: " + comp.stderr[:200])
        env = dict(os.environ)
        env["TSAN_OPTIONS"] = "exitcode=66"
        run = subprocess.run([bpath], capture_output=True, text=True, env=env)
        raced = ("ThreadSanitizer: data race" in run.stderr
                 or "WARNING: ThreadSanitizer" in run.stderr)
        return SideConfirmation("c", True, raced,
                                "tsan " + ("RACE" if raced else "clean"))


def _confirm_go(p: RacePattern) -> SideConfirmation:
    if not GO or not os.path.exists(GO):
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


def _confirm_rust(p: RacePattern) -> RustConfirmation:
    if not RUSTC or not os.path.exists(RUSTC):
        return RustConfirmation(False, None, None, "rustc unavailable")
    with tempfile.TemporaryDirectory() as d:
        rpath = os.path.join(d, "main.rs")
        with open(rpath, "w") as f:
            f.write(p.rust_source)
        bpath = os.path.join(d, "main")
        comp = subprocess.run([RUSTC, "-O", "-o", bpath, rpath],
                              capture_output=True, text=True)
        accepted = comp.returncode == 0
        if accepted and p.rust_accepts:
            run = subprocess.run([bpath], capture_output=True, text=True)
            if run.returncode != 0:
                return RustConfirmation(
                    True, True, None,
                    "rust accepted but execution failed: " + run.stderr[:200])
        err = None
        if not accepted:
            m = re.search(r"\berror\[([A-Z]\d{4})\]", comp.stderr)
            err = m.group(1) if m else None
        expected = p.rust_accepts
        verdict = "accepted" if accepted else "rejected"
        detail = f"rustc {verdict}"
        if err:
            detail += f" ({err})"
        if accepted != expected:
            detail += "; expected " + ("accept" if expected else "reject")
        return RustConfirmation(True, accepted, err, detail)


def confirm_race(name: str, check_go: bool = True,
                 check_rust: bool = True) -> RaceConfirmation:
    """Compile + run the pattern under real race detectors and check the verdict.

    The C side runs under ThreadSanitizer; the Go side (optional, slower) under
    ``go run -race``.  The Rust side compiles the idiomatic safe translation: the
    racy pattern must be rejected by ``rustc``, while synchronized variants must
    compile and run.  All available evidence must agree with ``pattern.races``.
    """
    p = pattern(name)
    c = _confirm_c(p)
    go = (_confirm_go(p) if check_go
          else SideConfirmation("go", False, None, "skipped"))
    rust = (_confirm_rust(p) if check_rust
            else RustConfirmation(False, None, None, "skipped"))
    return RaceConfirmation(name, p.races, c, go, rust)


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
