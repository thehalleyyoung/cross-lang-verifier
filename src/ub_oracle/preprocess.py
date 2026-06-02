"""Real C preprocessing (Step 26).

Actual C projects are not analysable as raw text: their meaning is fixed only
*after* the C preprocessor runs.  ``#include`` pulls in declarations, ``#define``
macros rewrite token streams (and the classic unparenthesized function-like macro
silently changes operator precedence), and ``#ifdef``/``#if`` conditionals select
which code even exists.  Any oracle that reasons about pre-preprocessing source is
reasoning about a program that does not exist — and macro-precedence surprises are
themselves a genuine divergence source a translation can get wrong.

This module integrates the **real** preprocessor (``clang -E``) and proves three
load-bearing facts against compiled, executed code:

  * **Macros are semantically load-bearing.**  ``#define MUL(a,b) a*b`` invoked as
    ``MUL(1+1, 2)`` expands to ``1+1*2`` and a real binary evaluates it to **3**,
    whereas the intended ``(1+1)*2`` is **4** — so an analyser that skips
    preprocessing (or mis-models the macro) would compare the wrong program.  We
    also detect such hazardous macros up front.
  * **Conditionals select the program.**  The same source preprocesses to
    different code under ``-DFEATURE`` vs not, and the selected branch is what
    actually runs.
  * **Includes resolve.**  A symbol defined only in an ``#include``d header is
    present after preprocessing and links/runs.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

CC = "/usr/bin/clang"


def preprocess(src: str,
               defines: Optional[List[str]] = None,
               include_dirs: Optional[List[str]] = None,
               keep_line_markers: bool = False) -> Optional[str]:
    """Run the real C preprocessor over ``src`` and return the expanded text.

    Returns ``None`` if clang is unavailable or preprocessing fails.
    """
    if not os.path.exists(CC):
        return None
    args = [CC, "-E"]
    if not keep_line_markers:
        args.append("-P")
    for d in (defines or []):
        args += ["-D", d]
    for inc in (include_dirs or []):
        args += ["-I", inc]
    with tempfile.TemporaryDirectory() as d:
        cpath = os.path.join(d, "a.c")
        with open(cpath, "w") as f:
            f.write(src)
        r = subprocess.run(args + [cpath], capture_output=True, text=True)
        if r.returncode != 0:
            return None
        return r.stdout


# --------------------------------------------------------------------------- #
# Hazardous-macro detection: a function-like macro whose body or whose parameter
# uses are not protected by parentheses can change precedence at the call site.
# --------------------------------------------------------------------------- #

_RE_FN_MACRO = re.compile(
    r"^[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)\(([^)]*)\)[ \t]+(.+?)[ \t]*$",
    re.MULTILINE)

# operators whose precedence can be subverted by an unparenthesized expansion.
_RE_BINOP = re.compile(r"[-+*/%]|<<|>>|[&|^]|&&|\|\||==|!=|[<>]=?")


@dataclass(frozen=True)
class MacroHazard:
    name: str
    params: Tuple[str, ...]
    body: str
    reason: str


def detect_unparenthesized_macros(src: str) -> List[MacroHazard]:
    """Flag function-like macros that risk precedence bugs: either the whole body
    is not wrapped in parentheses around a binary operator, or a parameter is used
    next to an operator without its own protective parentheses."""
    hazards: List[MacroHazard] = []
    for m in _RE_FN_MACRO.finditer(src):
        name, raw_params, body = m.group(1), m.group(2), m.group(3).strip()
        params = tuple(p.strip() for p in raw_params.split(",") if p.strip())
        # strip a trailing line continuation / comment noise.
        body_core = body
        has_binop = bool(_RE_BINOP.search(body_core))
        if not has_binop:
            continue
        whole_wrapped = body_core.startswith("(") and body_core.endswith(")") \
            and _balanced_outer(body_core)
        # is every parameter occurrence individually parenthesised?
        params_protected = all(_param_protected(body_core, p) for p in params)
        if not whole_wrapped or not params_protected:
            reason = []
            if not whole_wrapped:
                reason.append("body not fully parenthesised")
            if not params_protected:
                reason.append("a parameter is used without surrounding parens")
            hazards.append(MacroHazard(name, params, body_core,
                                       "; ".join(reason)))
    return hazards


def _balanced_outer(s: str) -> bool:
    """True if the outermost parentheses of ``s`` enclose the whole string."""
    if not (s.startswith("(") and s.endswith(")")):
        return False
    depth = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and i != len(s) - 1:
                return False
    return depth == 0


def _param_protected(body: str, param: str) -> bool:
    """True if every standalone occurrence of ``param`` is immediately wrapped in
    parentheses, i.e. appears as ``(param)``."""
    for m in re.finditer(r"\b" + re.escape(param) + r"\b", body):
        i, j = m.start(), m.end()
        before = body[i - 1] if i > 0 else ""
        after = body[j] if j < len(body) else ""
        if not (before == "(" and after == ")"):
            return False
    return True


# --------------------------------------------------------------------------- #
# Real-code confirmations.
# --------------------------------------------------------------------------- #

def _compile_run_int(full_src: str) -> Optional[int]:
    """Compile a complete C program that prints one integer; return that int."""
    if not os.path.exists(CC):
        return None
    with tempfile.TemporaryDirectory() as d:
        cpath = os.path.join(d, "a.c")
        with open(cpath, "w") as f:
            f.write(full_src)
        bpath = os.path.join(d, "a")
        comp = subprocess.run([CC, "-O0", "-o", bpath, cpath],
                              capture_output=True, text=True)
        if comp.returncode != 0:
            return None
        run = subprocess.run([bpath], capture_output=True, text=True)
        try:
            return int(run.stdout.strip())
        except ValueError:
            return None


HAZARD_MACRO = "#define MUL(a,b) a*b\n"
SAFE_MACRO = "#define MUL(a,b) ((a)*(b))\n"
_MACRO_CALL = "MUL(1+1, 2)"


@dataclass
class PreprocessConfirmation:
    available: bool
    expanded: str
    hazard_value: Optional[int]
    safe_value: Optional[int]
    detected_hazard: bool

    @property
    def ok(self) -> bool:
        # the hazard must expand to the precedence-bent value (3), the safe macro
        # to the intended value (4), and the detector must have flagged it.
        return (self.available and self.detected_hazard
                and self.hazard_value == 3 and self.safe_value == 4)


def confirm_macro_precedence_hazard() -> PreprocessConfirmation:
    """Prove a macro is semantically load-bearing: the unparenthesized form
    evaluates differently from the parenthesized one on a real binary, and the
    detector flags the hazardous form."""
    expanded = preprocess(HAZARD_MACRO + f"int v(){{ return {_MACRO_CALL}; }}\n")
    if expanded is None:
        return PreprocessConfirmation(False, "", None, None, False)
    main = "#include <stdio.h>\nint main(){{printf(\"%d\\n\", {});return 0;}}"
    hz = _compile_run_int(HAZARD_MACRO + main.format(_MACRO_CALL))
    sf = _compile_run_int(SAFE_MACRO + main.format(_MACRO_CALL))
    detected = any(h.name == "MUL"
                   for h in detect_unparenthesized_macros(HAZARD_MACRO))
    return PreprocessConfirmation(True, expanded.strip(), hz, sf, detected)


@dataclass
class ConditionalConfirmation:
    available: bool
    without: Optional[int]
    with_feature: Optional[int]

    @property
    def ok(self) -> bool:
        return (self.available and self.without == 0
                and self.with_feature == 1)


_COND_SRC = ("int sel(){\n#ifdef FEATURE\n return 1;\n#else\n return 0;\n"
             "#endif\n}\n")


def confirm_conditional_compilation() -> ConditionalConfirmation:
    """Prove ``#ifdef`` selects the program: the same source runs to a different
    value with and without ``-DFEATURE``."""
    if not os.path.exists(CC):
        return ConditionalConfirmation(False, None, None)
    main = "#include <stdio.h>\nint main(){printf(\"%d\\n\", sel());return 0;}"
    # compile the raw (un-pre-expanded) source directly with/without the define.
    def build(defines: List[str]) -> Optional[int]:
        with tempfile.TemporaryDirectory() as d:
            cpath = os.path.join(d, "a.c")
            with open(cpath, "w") as f:
                f.write(_COND_SRC + main)
            bpath = os.path.join(d, "a")
            args = [CC, "-O0"]
            for x in defines:
                args += ["-D", x]
            comp = subprocess.run(args + ["-o", bpath, cpath],
                                  capture_output=True, text=True)
            if comp.returncode != 0:
                return None
            run = subprocess.run([bpath], capture_output=True, text=True)
            try:
                return int(run.stdout.strip())
            except ValueError:
                return None
    return ConditionalConfirmation(True, build([]), build(["FEATURE"]))


@dataclass
class IncludeConfirmation:
    available: bool
    symbol_present: bool
    value: Optional[int]

    @property
    def ok(self) -> bool:
        return self.available and self.symbol_present and self.value == 42


def confirm_include_resolution() -> IncludeConfirmation:
    """Prove ``#include`` resolution: a macro defined only in a header is present
    after preprocessing and the program built from it runs."""
    if not os.path.exists(CC):
        return IncludeConfirmation(False, False, None)
    with tempfile.TemporaryDirectory() as d:
        hdr = os.path.join(d, "answer.h")
        with open(hdr, "w") as f:
            f.write("#ifndef ANSWER_H\n#define ANSWER_H\n#define ANSWER 42\n"
                    "#endif\n")
        src = '#include "answer.h"\nint get(){ return ANSWER; }\n'
        expanded = preprocess(src, include_dirs=[d])
        present = expanded is not None and "return 42" in expanded
        main = ('#include "answer.h"\n#include <stdio.h>\n'
                'int main(){printf("%d\\n", ANSWER);return 0;}')
        cpath = os.path.join(d, "m.c")
        with open(cpath, "w") as f:
            f.write(main)
        bpath = os.path.join(d, "m")
        comp = subprocess.run([CC, "-O0", "-I", d, "-o", bpath, cpath],
                              capture_output=True, text=True)
        val: Optional[int] = None
        if comp.returncode == 0:
            run = subprocess.run([bpath], capture_output=True, text=True)
            try:
                val = int(run.stdout.strip())
            except ValueError:
                val = None
        return IncludeConfirmation(True, present, val)
