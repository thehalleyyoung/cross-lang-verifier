"""
Target-semantics-pack conformance suite (100_STEPS step 124).

A :class:`~src.ub_oracle.target_semantics.TargetPack` is the *only* per-language
configuration the engine needs to support a new target (compiler invocation,
source suffix, the set of process return codes that count as a language-*defined*
outcome, and a data description of how each UB class is resolved).  Because the
whole generality claim rests on that one abstraction, a *new* pack that silently
violates the SPI contract — e.g. a non-total or value-only "defined" predicate, a
``compile_argv`` that forgets to wire the output path, a ``class_resolution`` that
names a divergence class that does not exist — would quietly poison the
re-execution harness (false "not confirmed", or worse, false "defined").

This module states the SPI obligations as executable **property checks** and runs
them against *every* registered pack, so adding a target language can never
regress the contract.  It is pure data — no compilers required — so it runs in the
fast CI gate.

The obligations (one :class:`Obligation` each):

  * ``name_matches_registry``    — the pack's ``name`` is its registry key.
  * ``suffix_is_a_dot_ext``      — ``source_suffix`` is a non-empty ``".ext"``.
  * ``compilers_declared``       — at least one candidate compiler is named.
  * ``defined_rc_well_formed``   — ``defined_returncodes`` is a non-empty tuple of
                                   ints that includes ``0`` (a returned value is
                                   *always* a defined outcome).
  * ``predicate_is_total``       — ``is_defined_returncode`` returns a ``bool`` for
                                   every probed return code and is ``True`` *iff*
                                   the code is in ``defined_returncodes``.
  * ``run_predicate_consistent`` — :meth:`RunOutcome.target_outcome_defined`
                                   agrees with the pack's data on every probe, and
                                   a timed-out run is never defined.
  * ``compile_argv_wires_io``    — ``compile_argv(cc, src, out)`` is a list that
                                   begins with the compiler and mentions both the
                                   source and the output path (so the harness's
                                   binary actually gets built where it looks).
  * ``compile_env_is_a_dict``    — ``compile_env(workdir)`` returns a ``dict`` of
                                   ``str -> str`` (a hermetic environment overlay).
  * ``resolutions_are_real``     — every ``class_resolution`` key is a real
                                   divergence-class key in the catalogue.

:func:`check_pack` returns one :class:`PackConformance` (the per-obligation
results); :func:`run_pack_conformance` runs the whole registry; and
:func:`confirm_pack_conformance` is the boolean merge gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

from .catalogue import CATALOGUE
from .reexec import RunOutcome
from .target_semantics import PACKS, TargetPack

#: a spread of process return codes every pack's predicate must classify total-ly:
#: success, common panic/abort codes, and Python's negative "killed by signal N".
_PROBE_RETURNCODES: Tuple[int, ...] = (
    0, 1, 2, 3, 5, 42, 101, 134, 139, -2, -4, -5, -6, -9, -11,
)


@dataclass(frozen=True)
class Obligation:
    """One SPI obligation a conforming pack must discharge."""

    key: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class PackConformance:
    """Per-pack conformance: the list of obligation results."""

    name: str
    obligations: Tuple[Obligation, ...]

    @property
    def ok(self) -> bool:
        return all(o.passed for o in self.obligations)

    def failures(self) -> List[Obligation]:
        return [o for o in self.obligations if not o.passed]


def _check_predicate_is_total(pack: TargetPack) -> Obligation:
    for rc in _PROBE_RETURNCODES:
        got = pack.is_defined_returncode(rc)
        if not isinstance(got, bool):
            return Obligation("predicate_is_total", False,
                              f"is_defined_returncode({rc}) returned non-bool {got!r}")
        if got != (rc in pack.defined_returncodes):
            return Obligation("predicate_is_total", False,
                              f"is_defined_returncode({rc})={got} disagrees with data")
    return Obligation("predicate_is_total", True,
                      f"total & data-consistent over {len(_PROBE_RETURNCODES)} probes")


def _check_run_predicate_consistent(pack: TargetPack) -> Obligation:
    for rc in _PROBE_RETURNCODES:
        defined = RunOutcome(rc, "out", "err").target_outcome_defined(pack.name)
        if defined != (rc in pack.defined_returncodes):
            return Obligation("run_predicate_consistent", False,
                              f"RunOutcome(rc={rc}).target_outcome_defined disagrees")
    # a timed-out run is never a defined outcome, regardless of return code.
    if RunOutcome(0, "", "", timed_out=True).target_outcome_defined(pack.name):
        return Obligation("run_predicate_consistent", False,
                          "a timed-out run was reported as defined")
    return Obligation("run_predicate_consistent", True,
                      "RunOutcome predicate matches pack data; timeout is non-defined")


def _check_compile_argv(pack: TargetPack) -> Obligation:
    src = "u" + pack.source_suffix
    out = "u.out"
    try:
        argv = pack.compile_argv("THE_CC", src, out)
    except Exception as e:  # pragma: no cover - defensive
        return Obligation("compile_argv_wires_io", False, f"raised {e!r}")
    if not isinstance(argv, list) or not argv:
        return Obligation("compile_argv_wires_io", False, f"not a non-empty list: {argv!r}")
    if argv[0] != "THE_CC":
        return Obligation("compile_argv_wires_io", False,
                          f"argv[0]={argv[0]!r} is not the compiler path")
    if not any(src in a for a in argv):
        return Obligation("compile_argv_wires_io", False, "source path never referenced")
    if not any(out in a for a in argv):
        return Obligation("compile_argv_wires_io", False, "output path never referenced")
    return Obligation("compile_argv_wires_io", True, " ".join(argv))


def _check_compile_env(pack: TargetPack) -> Obligation:
    try:
        env = pack.compile_env("/tmp/workdir")
    except Exception as e:  # pragma: no cover - defensive
        return Obligation("compile_env_is_a_dict", False, f"raised {e!r}")
    if not isinstance(env, dict):
        return Obligation("compile_env_is_a_dict", False, f"not a dict: {env!r}")
    for k, v in env.items():
        if not isinstance(k, str) or not isinstance(v, str):
            return Obligation("compile_env_is_a_dict", False,
                              f"non-str entry {k!r}={v!r}")
    return Obligation("compile_env_is_a_dict", True, f"{len(env)} hermetic var(s)")


def _check_resolutions_are_real(pack: TargetPack) -> Obligation:
    unknown = [k for k in pack.class_resolution if k not in CATALOGUE]
    if unknown:
        return Obligation("resolutions_are_real", False,
                          f"class_resolution names unknown classes: {unknown}")
    return Obligation("resolutions_are_real", True,
                      f"{len(pack.class_resolution)} resolution(s), all in catalogue")


def check_pack(name: str) -> PackConformance:
    """Run every SPI obligation against the named pack."""
    pack = PACKS[name]
    obs: List[Obligation] = []

    obs.append(Obligation("name_matches_registry", pack.name == name,
                          f"pack.name={pack.name!r} key={name!r}"))

    suf = pack.source_suffix
    obs.append(Obligation("suffix_is_a_dot_ext",
                          isinstance(suf, str) and len(suf) >= 2 and suf.startswith("."),
                          f"source_suffix={suf!r}"))

    obs.append(Obligation("compilers_declared",
                          bool(pack.compiler_candidates)
                          and all(isinstance(c, str) and c
                                  for c in pack.compiler_candidates),
                          f"candidates={list(pack.compiler_candidates)}"))

    drc = pack.defined_returncodes
    obs.append(Obligation(
        "defined_rc_well_formed",
        isinstance(drc, tuple) and len(drc) >= 1
        and all(isinstance(c, int) for c in drc) and 0 in drc,
        f"defined_returncodes={drc}"))

    obs.append(_check_predicate_is_total(pack))
    obs.append(_check_run_predicate_consistent(pack))
    obs.append(_check_compile_argv(pack))
    obs.append(_check_compile_env(pack))
    obs.append(_check_resolutions_are_real(pack))

    return PackConformance(name, tuple(obs))


#: the SPI obligations every pack is checked against (stable, for tests/docs).
OBLIGATION_KEYS: Tuple[str, ...] = (
    "name_matches_registry",
    "suffix_is_a_dot_ext",
    "compilers_declared",
    "defined_rc_well_formed",
    "predicate_is_total",
    "run_predicate_consistent",
    "compile_argv_wires_io",
    "compile_env_is_a_dict",
    "resolutions_are_real",
)


def run_pack_conformance() -> Dict[str, PackConformance]:
    """Run the conformance suite against every registered pack."""
    return {name: check_pack(name) for name in PACKS}


@dataclass(frozen=True)
class PackConformanceConfirmation:
    by_pack: Dict[str, PackConformance]

    @property
    def ok(self) -> bool:
        return all(pc.ok for pc in self.by_pack.values())

    @property
    def n_packs(self) -> int:
        return len(self.by_pack)

    def detail(self) -> str:
        bad = {n: [o.key for o in pc.failures()]
               for n, pc in self.by_pack.items() if not pc.ok}
        if not bad:
            return f"all {self.n_packs} packs satisfy the SPI contract"
        return f"SPI conformance failures: {bad}"


def confirm_pack_conformance() -> PackConformanceConfirmation:
    """The merge gate: every pack must discharge every SPI obligation."""
    return PackConformanceConfirmation(run_pack_conformance())
