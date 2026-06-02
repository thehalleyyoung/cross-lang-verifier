"""
``longjmp`` to an exited VLA scope divergence oracle (100_STEPS step 111).

This class is easy to get subtly wrong.  C does **not** make every jump out of a
VLA scope undefined: a ``setjmp`` outside a VLA block followed by ``longjmp`` out
of the block is a permitted leak hazard.  The undefined case is the reverse
control-flow edge from C17 7.13.2.1: ``setjmp`` is evaluated while a variably
modified object is in scope, that scope is left, and a later ``longjmp`` targets
the saved context.  The jump re-enters a dead VLA scope.

UBSan does not instrument this precondition, so the confirmation path uses the
same executable-contract style as the ``memcpy`` overlap oracle: a real C binary
is compiled with ``-DCLV_CHECK_LONGJMP_VLA`` and tracks the VLA depth captured by
``setjmp``.  If ``longjmp`` targets a context whose captured VLA depth is no
longer live, it aborts with a stable ``longjmp-vla`` diagnostic.  The target
programs are ordinary safe Rust/Go programs that perform the same exceptional
control transfer through structured unwinding, proving cleanup runs and the
outcome is deterministic.
"""

from __future__ import annotations

from typing import Dict, Optional

import z3

from ..catalogue import LONGJMP_VLA, Definedness
from ..plugin import DivergenceOracle, OracleResult, OracleVerdict, register
from ..reexec import ReexecHarness
from ..replay import Counterexample

_CONTRACT_MACRO = "CLV_CHECK_LONGJMP_VLA"
_CONTRACT_TOKEN = "longjmp-vla"


def _find_positive_bound(unit: Dict, var: str) -> Optional[int]:
    """Pick a clean positive VLA bound, honoring an optional declared range.

    The UB trigger is the stale ``jmp_buf`` target, not an invalid VLA bound, so
    the witness deliberately keeps the VLA size positive to isolate this class
    from the VLA-bound oracle.
    """
    preferred = int(unit.get("preferred_bound", 4))
    rng = unit.get("bound_range")
    if rng is None:
        return preferred if preferred > 0 else None

    n = z3.Int(var)
    opt = z3.Optimize()
    opt.add(n >= int(rng[0]), n <= int(rng[1]), n > 0)
    opt.minimize(n)
    if opt.check() != z3.sat:
        return None
    return opt.model().eval(n, model_completion=True).as_long()


def _c_longjmp_vla_program(var: str) -> str:
    return (
        "#include <setjmp.h>\n"
        "#include <stdio.h>\n"
        "#include <stdlib.h>\n"
        "\n"
        "static jmp_buf jb;\n"
        "static volatile int sink;\n"
        "\n"
        "#ifdef CLV_CHECK_LONGJMP_VLA\n"
        "static int clv_vla_depth;\n"
        "static int clv_setjmp_vla_depth;\n"
        "static void clv_enter_vla(void) { clv_vla_depth += 1; }\n"
        "static void clv_leave_vla(void) { clv_vla_depth -= 1; }\n"
        "static void clv_capture_setjmp(void) { clv_setjmp_vla_depth = clv_vla_depth; }\n"
        "#define CLV_SETJMP(env) (clv_capture_setjmp(), setjmp(env))\n"
        "static void clv_longjmp(jmp_buf env, int value) {\n"
        "    if (clv_setjmp_vla_depth > clv_vla_depth) {\n"
        "        fputs(\"runtime error: clv-contract: longjmp-vla target has exited VLA scope\\n\", stderr);\n"
        "        abort();\n"
        "    }\n"
        "    longjmp(env, value);\n"
        "}\n"
        "#else\n"
        "#define clv_enter_vla() ((void)0)\n"
        "#define clv_leave_vla() ((void)0)\n"
        "#define CLV_SETJMP(env) setjmp(env)\n"
        "#define clv_longjmp(env, value) longjmp((env), (value))\n"
        "#endif\n"
        "\n"
        f"static int f(int {var}) {{\n"
        f"    if ({var} <= 0) return 2;\n"
        "    {\n"
        f"        int a[{var}];\n"
        "        clv_enter_vla();\n"
        f"        for (int i = 0; i < {var}; ++i) a[i] = i + 1;\n"
        f"        sink = a[{var} - 1];\n"
        "        if (CLV_SETJMP(jb) != 0) {\n"
        "            return sink + 100;\n"
        "        }\n"
        "        clv_leave_vla();\n"
        "    }\n"
        "    clv_longjmp(jb, 1);\n"
        "    return -1;\n"
        "}\n"
        "\n"
        "int main(int argc, char **argv) {\n"
        "    if (argc < 2) return 2;\n"
        f"    int {var} = (int)strtol(argv[1], 0, 10);\n"
        f"    printf(\"%d\\n\", f({var}));\n"
        "    return 0;\n"
        "}\n"
    )


def leak_only_c_control_program(var: str = "n") -> str:
    """A negative-control C program: ``longjmp`` leaves no dead VLA target.

    ``setjmp`` is outside the VLA block, so the later ``longjmp`` does not target
    a context captured inside a VLA scope.  Implementations may leak stack VLA
    storage on such paths, but this is not the C17 7.13.2.1 UB modeled here.
    """
    return (
        "#include <setjmp.h>\n"
        "#include <stdio.h>\n"
        "#include <stdlib.h>\n"
        "\n"
        "static jmp_buf jb;\n"
        "static volatile int sink;\n"
        "\n"
        "#ifdef CLV_CHECK_LONGJMP_VLA\n"
        "static int clv_vla_depth;\n"
        "static int clv_setjmp_vla_depth;\n"
        "static void clv_enter_vla(void) { clv_vla_depth += 1; }\n"
        "static void clv_leave_vla(void) { clv_vla_depth -= 1; }\n"
        "static void clv_capture_setjmp(void) { clv_setjmp_vla_depth = clv_vla_depth; }\n"
        "#define CLV_SETJMP(env) (clv_capture_setjmp(), setjmp(env))\n"
        "static void clv_longjmp(jmp_buf env, int value) {\n"
        "    if (clv_setjmp_vla_depth > clv_vla_depth) {\n"
        "        fputs(\"runtime error: clv-contract: longjmp-vla target has exited VLA scope\\n\", stderr);\n"
        "        abort();\n"
        "    }\n"
        "    longjmp(env, value);\n"
        "}\n"
        "#else\n"
        "#define clv_enter_vla() ((void)0)\n"
        "#define clv_leave_vla() ((void)0)\n"
        "#define CLV_SETJMP(env) setjmp(env)\n"
        "#define clv_longjmp(env, value) longjmp((env), (value))\n"
        "#endif\n"
        "\n"
        f"static int f(int {var}) {{\n"
        f"    if ({var} <= 0) return 2;\n"
        "    if (CLV_SETJMP(jb) != 0) {\n"
        "        return sink + 100;\n"
        "    }\n"
        "    {\n"
        f"        int a[{var}];\n"
        "        clv_enter_vla();\n"
        f"        for (int i = 0; i < {var}; ++i) a[i] = i + 1;\n"
        f"        sink = a[{var} - 1];\n"
        "        clv_leave_vla();\n"
        "    }\n"
        "    clv_longjmp(jb, 1);\n"
        "    return -1;\n"
        "}\n"
        "\n"
        "int main(int argc, char **argv) {\n"
        "    if (argc < 2) return 2;\n"
        f"    int {var} = (int)strtol(argv[1], 0, 10);\n"
        f"    printf(\"%d\\n\", f({var}));\n"
        "    return 0;\n"
        "}\n"
    )


def _rust_structured_unwind_program(var: str) -> str:
    return (
        "use std::cell::Cell;\n"
        "use std::panic::{self, AssertUnwindSafe};\n"
        "\n"
        "struct Guard<'a>(&'a Cell<i32>);\n"
        "impl<'a> Drop for Guard<'a> {\n"
        "    fn drop(&mut self) { self.0.set(self.0.get() + 100); }\n"
        "}\n"
        "\n"
        f"fn f({var}: i32) -> i32 {{\n"
        "    let cleaned = Cell::new(0);\n"
        "    let result = panic::catch_unwind(AssertUnwindSafe(|| {\n"
        "        let _guard = Guard(&cleaned);\n"
        f"        if {var} <= 0 {{ panic!(\"positive VLA witness required\"); }}\n"
        f"        let len = {var} as usize;\n"
        "        let mut a = vec![0i32; len];\n"
        "        for i in 0..len { a[i] = i as i32 + 1; }\n"
        "        let _sink = a[len - 1];\n"
        "        panic!(\"structured unwind\");\n"
        "    }));\n"
        "    assert!(result.is_err());\n"
        f"    cleaned.get() + {var}\n"
        "}\n"
        "\n"
        "fn main() {\n"
        f"    let {var}: i32 = std::env::args().nth(1).unwrap().parse().unwrap();\n"
        f"    println!(\"{{}}\", f({var}));\n"
        "}\n"
    )


def _go_structured_unwind_program(var: str) -> str:
    return (
        "package main\n"
        "\n"
        "import (\n"
        "\t\"fmt\"\n"
        "\t\"os\"\n"
        "\t\"strconv\"\n"
        ")\n"
        "\n"
        f"func f({var} int) (out int) {{\n"
        "\tcleaned := 0\n"
        "\tdefer func() {\n"
        "\t\tif recover() != nil {\n"
        f"\t\t\tout = cleaned + {var}\n"
        "\t\t}\n"
        "\t}()\n"
        "\tdefer func() { cleaned += 100 }()\n"
        f"\tif {var} <= 0 {{ panic(\"positive VLA witness required\") }}\n"
        f"\ta := make([]int, {var})\n"
        f"\tfor i := 0; i < {var}; i++ {{ a[i] = i + 1 }}\n"
        f"\t_ = a[{var}-1]\n"
        "\tpanic(\"structured unwind\")\n"
        "}\n"
        "\n"
        "func main() {\n"
        f"\t{var}, _ := strconv.Atoi(os.Args[1])\n"
        f"\tfmt.Println(f({var}))\n"
        "}\n"
    )


class _LongjmpVlaBase(DivergenceOracle):
    divergence_class = LONGJMP_VLA.key
    source_lang = "c"
    confirmation_mode = "libc_contract_trap_vs_defined"

    def applies_to(self, unit: Dict) -> bool:
        if unit.get("probe") not in (None, self.divergence_class):
            return False
        return unit.get("kind") in {"longjmp_vla", "setjmp_longjmp_vla"}

    def _result(self, unit: Dict, target_src: str) -> OracleResult:
        if not self.applies_to(unit):
            return OracleResult(OracleVerdict.NOT_APPLICABLE, self.divergence_class,
                                detail="unit is not a longjmp-to-exited-VLA pattern")
        var = unit.get("var", "n")
        n_val = _find_positive_bound(unit, var)
        if n_val is None:
            return OracleResult(OracleVerdict.NO_DIVERGENCE_FOUND,
                                self.divergence_class,
                                detail="no positive VLA bound in declared range")
        ce = self._build(var, n_val, target_src)
        return OracleResult(
            OracleVerdict.DIVERGENT,
            self.divergence_class,
            counterexample=ce,
            detail=f"witness {var}={n_val}; setjmp target is inside an exited VLA scope",
        )

    def _build(self, var: str, n_val: int, target_src: str) -> Counterexample:
        return Counterexample(
            divergence_class=self.divergence_class,
            source_lang="c",
            target_lang=self.target_lang,
            inputs={var: n_val},
            source_snippet=_c_longjmp_vla_program(var),
            target_snippet=target_src,
            source_definedness=Definedness.UNDEFINED.value,
            divergence_witness=(
                f"C evaluates `setjmp` while a positive-bound VLA (`int a[{var}]`, "
                f"{var}={n_val}) is in scope, leaves that block, then calls "
                f"`longjmp` to the saved context. C17 7.13.2.1 makes this "
                f"undefined because the target context's variably-modified scope "
                f"has been exited; the checked contract build reports "
                f"`{_CONTRACT_TOKEN}`. The safe {self.target_lang} port expresses "
                f"the same nonlocal control transfer through structured unwinding, "
                f"so cleanup runs and the outcome is deterministic."
            ),
            definedness_witness=(
                f"{var}={n_val} is a positive, valid VLA bound; the UB is solely "
                f"the stale `jmp_buf` target into an exited VLA scope, not an "
                f"invalid array bound."
            ),
        )

    def confirm(self, result: OracleResult,
                harness: Optional[ReexecHarness] = None) -> OracleResult:
        if result.counterexample is None or not result.is_divergent:
            return result
        ce = result.counterexample
        harness = harness or ReexecHarness()
        argv = [str(v) for v in ce.inputs.values()]
        rr = harness.confirm_libc_contract_trap_vs_defined(
            ce.source_snippet,
            ce.target_snippet,
            argv,
            ce.divergence_class,
            target_lang=self.target_lang,
            contract_macro=_CONTRACT_MACRO,
            contract_token=_CONTRACT_TOKEN,
            use_asan=False,
        )
        result.reexec = rr
        if rr.available:
            ce.confirmed = rr.confirmed
            ce.source_observed = {k: v.stdout for k, v in rr.c_runs.items()}
            ce.target_observed = rr.rust_run.stdout if rr.rust_run else None
        return result


class LongjmpVlaOracle(_LongjmpVlaBase):
    """C ``longjmp`` into an exited VLA scope vs Rust structured unwinding."""

    target_lang = "rust"

    def find_divergence(self, unit: Dict) -> OracleResult:
        var = unit.get("var", "n")
        return self._result(unit, _rust_structured_unwind_program(var))


class GoLongjmpVlaOracle(_LongjmpVlaBase):
    """C ``longjmp`` into an exited VLA scope vs Go ``defer``/``recover``."""

    target_lang = "go"

    def find_divergence(self, unit: Dict) -> OracleResult:
        var = unit.get("var", "n")
        return self._result(unit, _go_structured_unwind_program(var))


register(LongjmpVlaOracle())
register(GoLongjmpVlaOracle())
