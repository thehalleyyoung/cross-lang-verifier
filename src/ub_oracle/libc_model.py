"""Behavior-accurate libc / runtime modeling (Step 32).

Real C code is saturated with calls into the C runtime — ``mem*``, ``str*``,
allocation, a little math.  An oracle that treats these as opaque cannot reason
about the programs that actually ship; one that models them *wrongly* is worse
than useless.  This module provides **behavior-accurate, executable specs** for
the most-hit libc surface and a **reusable confirmation framework** that proves
each spec against the *real* libc on randomized inputs: every spec is run on the
host's libc through a compiled harness and its output compared to the model.

Specs are pure Python over ``bytes``/``int`` and are deliberately precise about
the runtime's value contract — e.g. ``strcmp``/``memcmp`` return only the *sign*
of the first differing (unsigned) byte, ``strlen`` counts up to the terminating
NUL, ``memmove`` is defined under overlap while ``memcpy`` is not.  The framework
(``LibcSpec`` + ``confirm_spec``) is runtime-agnostic: a new function is a model
function plus a harness mode, and the same randomized-differential check applies.
"""

from __future__ import annotations

import os
import random
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

CC = "/usr/bin/clang"


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


# --------------------------------------------------------------------------- #
# Executable specs (pure Python).  Inputs use Python bytes; a trailing NUL is
# included where the libc function requires NUL-termination.
# --------------------------------------------------------------------------- #

def model_strlen(s: bytes) -> int:
    i = s.find(b"\x00")
    if i < 0:
        raise ValueError("strlen requires a NUL terminator (UB otherwise)")
    return i


def model_strcmp(a: bytes, b: bytes) -> int:
    ai = a.find(b"\x00")
    bi = b.find(b"\x00")
    if ai < 0 or bi < 0:
        raise ValueError("strcmp requires NUL-terminated inputs")
    ca, cb = a[:ai + 1], b[:bi + 1]
    for x, y in zip(ca, cb):
        if x != y:
            return _sign(x - y)
    return 0


def model_strncmp(a: bytes, b: bytes, n: int) -> int:
    for k in range(n):
        x = a[k] if k < len(a) else 0
        y = b[k] if k < len(b) else 0
        if x != y:
            return _sign(x - y)
        if x == 0:  # both NUL -> equal, stop
            return 0
    return 0


def model_memcmp(a: bytes, b: bytes, n: int) -> int:
    for k in range(n):
        if a[k] != b[k]:
            return _sign(a[k] - b[k])
    return 0


def model_memcpy(src: bytes, n: int) -> bytes:
    # result of copying n bytes of src into a fresh buffer.
    return bytes(src[:n])


def model_memset(buf: bytes, c: int, n: int) -> bytes:
    out = bytearray(buf)
    for k in range(n):
        out[k] = c & 0xFF
    return bytes(out)


def model_strchr(s: bytes, c: int) -> int:
    # returns the index of the first occurrence of c (as char) in s up to and
    # including the NUL terminator, or -1 if absent.
    end = s.find(b"\x00")
    if end < 0:
        raise ValueError("strchr requires a NUL terminator")
    target = c & 0xFF
    hay = s[:end + 1]
    idx = hay.find(bytes([target]))
    return idx if idx >= 0 else -1


# --------------------------------------------------------------------------- #
# Reusable confirmation framework.
# --------------------------------------------------------------------------- #

# C harness: one binary, many modes; buffers are passed as hex on argv.
_HARNESS = r"""
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static int from_hex_nibble(char c){
    if(c>='0'&&c<='9') return c-'0';
    if(c>='a'&&c<='f') return c-'a'+10;
    if(c>='A'&&c<='F') return c-'A'+10;
    return 0;
}
static size_t unhex(const char*h, unsigned char*out, size_t cap){
    size_t n=0; for(size_t i=0; h[i] && h[i+1] && n<cap; i+=2)
        out[n++] = (unsigned char)((from_hex_nibble(h[i])<<4)|from_hex_nibble(h[i+1]));
    return n;
}
static void put_hex(const unsigned char*b, size_t n){
    for(size_t i=0;i<n;i++) printf("%02x", b[i]); printf("\n");
}

int main(int argc,char**argv){
    const char*mode=argv[1];
    if(!strcmp(mode,"strlen")){
        unsigned char a[4096]; size_t n=unhex(argv[2],a,sizeof a);
        printf("%zu\n", strlen((char*)a)); (void)n;
    } else if(!strcmp(mode,"strcmp")){
        unsigned char a[4096],b[4096]; unhex(argv[2],a,sizeof a); unhex(argv[3],b,sizeof b);
        printf("%d\n", (strcmp((char*)a,(char*)b)>0)-(strcmp((char*)a,(char*)b)<0));
    } else if(!strcmp(mode,"strncmp")){
        unsigned char a[4096],b[4096]; unhex(argv[2],a,sizeof a); unhex(argv[3],b,sizeof b);
        int n=atoi(argv[4]); int r=strncmp((char*)a,(char*)b,(size_t)n);
        printf("%d\n",(r>0)-(r<0));
    } else if(!strcmp(mode,"memcmp")){
        unsigned char a[4096],b[4096]; unhex(argv[2],a,sizeof a); unhex(argv[3],b,sizeof b);
        int n=atoi(argv[4]); int r=memcmp(a,b,(size_t)n);
        printf("%d\n",(r>0)-(r<0));
    } else if(!strcmp(mode,"memcpy")){
        unsigned char s[4096],dst[4096]; size_t sn=unhex(argv[2],s,sizeof s);
        int n=atoi(argv[3]); memset(dst,0,sizeof dst); memcpy(dst,s,(size_t)n);
        put_hex(dst,(size_t)n); (void)sn;
    } else if(!strcmp(mode,"memset")){
        unsigned char buf[4096]; size_t bn=unhex(argv[2],buf,sizeof buf);
        int c=atoi(argv[3]); int n=atoi(argv[4]); memset(buf,c,(size_t)n);
        put_hex(buf,bn);
    } else if(!strcmp(mode,"strchr")){
        unsigned char a[4096]; unhex(argv[2],a,sizeof a); int c=atoi(argv[3]);
        char*p=strchr((char*)a,c);
        printf("%ld\n", p? (long)(p-(char*)a) : -1L);
    }
    return 0;
}
"""


def _hx(b: bytes) -> str:
    return b.hex()


@dataclass
class LibcSpec:
    name: str
    mode: str
    model: Callable[..., object]
    # produce (model_args, argv_extra) for a random case.
    gen: Callable[[random.Random], Tuple[tuple, List[str]]]
    # format the model's result to match harness stdout.
    fmt: Callable[[object], str]


def _gen_str(rng: random.Random, maxlen: int = 12) -> bytes:
    n = rng.randint(0, maxlen)
    body = bytes(rng.randint(1, 255) for _ in range(n))  # no embedded NUL
    return body + b"\x00"


def _two_strs(rng: random.Random) -> Tuple[bytes, bytes]:
    a = _gen_str(rng)
    if rng.random() < 0.4:  # sometimes make them share a prefix
        k = rng.randint(0, len(a) - 1)
        b = a[:k] + _gen_str(rng)
    else:
        b = _gen_str(rng)
    return a, b


SPECS: Dict[str, LibcSpec] = {
    "strlen": LibcSpec(
        "strlen", "strlen", model_strlen,
        lambda rng: ((_gen_str(rng),), []),
        lambda r: str(r)),
    "strcmp": LibcSpec(
        "strcmp", "strcmp", model_strcmp,
        lambda rng: (lambda a, b: ((a, b), [_hx(b)]))(*_two_strs(rng)),
        lambda r: str(r)),
    "strncmp": LibcSpec(
        "strncmp", "strncmp", model_strncmp,
        lambda rng: (lambda a, b, n: ((a, b, n), [_hx(b), str(n)]))(
            *(_two_strs(rng) + (rng.randint(0, 8),))),
        lambda r: str(r)),
    "memcmp": LibcSpec(
        "memcmp", "memcmp", model_memcmp,
        lambda rng: (lambda a, b, n: ((a, b, n), [_hx(b), str(n)]))(
            *_mem_pair(rng)),
        lambda r: str(r)),
    "memcpy": LibcSpec(
        "memcpy", "memcpy", model_memcpy,
        lambda rng: (lambda s, n: ((s, n), [str(n)]))(*_mem_copy_case(rng)),
        lambda r: r.hex()),  # type: ignore[union-attr]
    "memset": LibcSpec(
        "memset", "memset", model_memset,
        lambda rng: (lambda buf, c, n: ((buf, c, n), [str(c), str(n)]))(
            *_mem_set_case(rng)),
        lambda r: r.hex()),  # type: ignore[union-attr]
    "strchr": LibcSpec(
        "strchr", "strchr", model_strchr,
        lambda rng: (lambda s, c: ((s, c), [str(c)]))(*_strchr_case(rng)),
        lambda r: str(r)),
}


def _mem_pair(rng: random.Random) -> Tuple[bytes, bytes, int]:
    n = rng.randint(1, 16)
    a = bytes(rng.randint(0, 255) for _ in range(n))
    if rng.random() < 0.5:
        b = bytearray(a)
        if n:
            i = rng.randint(0, n - 1)
            b[i] = (b[i] + rng.randint(1, 255)) & 0xFF
        b = bytes(b)
    else:
        b = bytes(rng.randint(0, 255) for _ in range(n))
    return a, b, n


def _mem_copy_case(rng: random.Random) -> Tuple[bytes, int]:
    n = rng.randint(1, 16)
    s = bytes(rng.randint(0, 255) for _ in range(n))
    return s, n


def _mem_set_case(rng: random.Random) -> Tuple[bytes, int, int]:
    blen = rng.randint(1, 16)
    buf = bytes(rng.randint(0, 255) for _ in range(blen))
    n = rng.randint(0, blen)
    c = rng.randint(0, 255)
    return buf, c, n


def _strchr_case(rng: random.Random) -> Tuple[bytes, int]:
    s = _gen_str(rng)
    # half the time pick a char that's present.
    if rng.random() < 0.6 and len(s) > 1:
        c = s[rng.randint(0, len(s) - 2)]
    else:
        c = rng.randint(0, 255)
    return s, c


@dataclass
class SpecConfirmation:
    name: str
    available: bool
    trials: int
    mismatches: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.available and self.trials > 0 and not self.mismatches


def _build_harness(tmpdir: str) -> Optional[str]:
    if not os.path.exists(CC):
        return None
    cpath = os.path.join(tmpdir, "h.c")
    with open(cpath, "w") as f:
        f.write(_HARNESS)
    bpath = os.path.join(tmpdir, "h")
    comp = subprocess.run([CC, "-O0", "-o", bpath, cpath],
                          capture_output=True, text=True)
    if comp.returncode != 0:
        return None
    return bpath


def confirm_spec(name: str, trials: int = 200,
                 seed: int = 1234) -> SpecConfirmation:
    """Randomized differential check of one spec against the real libc."""
    spec = SPECS[name]
    if not os.path.exists(CC):
        return SpecConfirmation(name, False, 0)
    rng = random.Random(seed)
    with tempfile.TemporaryDirectory() as d:
        harness = _build_harness(d)
        if harness is None:
            return SpecConfirmation(name, False, 0)
        mismatches: List[str] = []
        for _ in range(trials):
            model_args, argv_extra = spec.gen(rng)
            # argv[2] is always the (hex of the) first buffer/string arg.
            first = model_args[0]
            assert isinstance(first, (bytes, bytearray))
            argv = [harness, spec.mode, first.hex()] + argv_extra
            run = subprocess.run(argv, capture_output=True, text=True)
            real = run.stdout.strip()
            try:
                model_out = spec.fmt(spec.model(*model_args))
            except ValueError:
                continue  # UB precondition not met; skip (not part of contract)
            if real != model_out:
                mismatches.append(
                    f"args={model_args!r} real={real!r} model={model_out!r}")
                if len(mismatches) > 8:
                    break
        return SpecConfirmation(name, True, trials, mismatches)


def confirm_all(trials: int = 120) -> List[SpecConfirmation]:
    return [confirm_spec(n, trials=trials) for n in SPECS]


# UB / value contracts the specs encode, surfaced for documentation and so the
# oracle can abstain when a call site violates a precondition.
LIBC_CONTRACTS: Dict[str, str] = {
    "strlen": "argument must be NUL-terminated; reads until the first NUL.",
    "strcmp": "both arguments NUL-terminated; returns only the SIGN of the "
              "first differing unsigned-char pair.",
    "memcmp": "compares exactly n bytes; returns the SIGN of the first "
              "differing unsigned-char pair.",
    "memcpy": "source and destination must NOT overlap (UB if they do); copies "
              "exactly n bytes.",
    "memmove": "defined even when regions overlap (unlike memcpy).",
    "memset": "fills exactly n bytes with (unsigned char)c.",
    "strchr": "argument NUL-terminated; the terminating NUL is part of the "
              "searched range.",
}
