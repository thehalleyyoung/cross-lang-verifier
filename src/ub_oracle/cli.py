"""
Pair-aware command-line verifier (100_STEPS step 55).

``cross-lang-verify`` reads a manifest of translation *units* — each a typed
description of a ``(source, target)`` fragment plus its declared language pair —
runs every applicable divergence oracle on each, and prints a colored, actionable
summary plus an honest abstention breakdown (Step 47). Confirmed divergences can
be exported as SARIF (Step 57) for GitHub code scanning.

Honesty contract: the engine is *pair-aware*, not magically pair-agnostic. Only
the language pairs with a registered oracle (C->Rust today) can be decided; a
unit declaring any other pair gets a loud ``NOT_COVERED`` verdict, and the
summary surfaces the uncovered count so missing support is never hidden. The
tool is *sound for divergence*: it never claims equivalence.

Manifest format (JSON)::

    {"units": [
       {"name": "add1", "kind": "binop_const", "op": "add", "const": 1,
        "width": 32, "var": "x", "signed": true, "probe": "signed_overflow",
        "source_lang": "c", "target_lang": "rust"},
       ...
    ]}

A bare top-level JSON list of unit objects is also accepted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

from .report import aggregate_reports, pair_of, to_sarif
from .reexec import toolchain_available
from .cache import VerificationCache, verify_incremental
from .dashboard import render_dashboard
from .suppress import (
    apply_suppressions,
    build_baseline,
    load_suppressions,
)
from .triage import render_triage, triage_reports
from .verify import VerifyReport, VerifyVerdict, verify_unit
from .ir import assert_valid, IRValidationError

_VERDICT_STYLE = {
    VerifyVerdict.DIVERGENT: ("31;1", "DIVERGENT"),          # bold red
    VerifyVerdict.CANDIDATE: ("33;1", "CANDIDATE"),          # bold yellow
    VerifyVerdict.NO_DIVERGENCE_FOUND: ("32", "NO-DIVERGENCE"),  # green
    VerifyVerdict.UNKNOWN: ("35", "UNKNOWN"),                # magenta
    VerifyVerdict.NOT_COVERED: ("90", "NOT-COVERED"),        # bright black
}

#: which verdicts each --fail-on token selects.
_FAIL_ON = {
    "divergent": {VerifyVerdict.DIVERGENT},
    "candidate": {VerifyVerdict.CANDIDATE},
    "unknown": {VerifyVerdict.UNKNOWN},
    "not-covered": {VerifyVerdict.NOT_COVERED},
}


class _Color:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def __call__(self, code: str, text: str) -> str:
        if not self.enabled:
            return text
        return f"\033[{code}m{text}\033[0m"


def _want_color(arg: str) -> bool:
    if arg == "always":
        return True
    if arg == "never":
        return False
    # auto
    if os.environ.get("NO_COLOR") is not None:
        return False
    return sys.stdout.isatty()


def _load_units(path: str, *, validate: bool = True) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        units = data.get("units")
        if units is None:
            raise ValueError("manifest object must have a 'units' array")
    elif isinstance(data, list):
        units = data
    else:
        raise ValueError("manifest must be a JSON object or array")
    if not isinstance(units, list) or not all(isinstance(u, dict) for u in units):
        raise ValueError("'units' must be an array of unit objects")
    if validate:
        # Reject ill-formed lowerings loudly at the frontend boundary (step 6):
        # every unit must satisfy the frozen shared-IR contract before the
        # engine touches it.
        for i, unit in enumerate(units):
            assert_valid(unit, label=_unit_label(unit, i))
    return units


def _unit_label(unit: Dict, i: int) -> str:
    return str(unit.get("name") or unit.get("id") or f"unit[{i}]")


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cross-lang-verify",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Verify a manifest of cross-language translation units for "
            "semantic divergence.\n\n"
            "The tool is SOUND FOR DIVERGENCE: a DIVERGENT verdict is confirmed "
            "by really compiling and running the source and target; it NEVER "
            "claims the two are equivalent. NO-DIVERGENCE means only the covered "
            "classes were checked."),
        epilog=(
            "Examples:\n"
            "  cross-lang-verify --units units.json\n"
            "  cross-lang-verify --units units.json --triage\n"
            "  cross-lang-verify --units units.json --sarif out.sarif\n"
            "  cross-lang-verify --units units.json --format json --no-confirm\n"
            "  cross-lang-verify --units units.json --write-baseline baseline.json\n"
            "  cross-lang-verify --units units.json --suppress baseline.json\n"
            "  cross-lang-verify --units units.json --fail-on divergent "
            "--fail-on candidate\n"),
    )
    p.add_argument("--units", required=True, metavar="MANIFEST.json",
                   help="path to the JSON units manifest")
    p.add_argument("--sarif", metavar="OUT.sarif",
                   help="also write a SARIF 2.1.0 log of the findings")
    p.add_argument("--format", choices=("text", "json"), default="text",
                   help="stdout format for the summary (default: text)")
    p.add_argument("--color", choices=("auto", "always", "never"), default="auto",
                   help="colorize text output (default: auto)")
    p.add_argument("--no-confirm", action="store_true",
                   help="skip ground-truth re-execution; symbolic witnesses are "
                        "reported as CANDIDATE, never as DIVERGENT")
    p.add_argument("--fail-on", action="append", default=None,
                   choices=sorted(_FAIL_ON),
                   help="exit nonzero if any unit gets this verdict (repeatable; "
                        "default: divergent)")
    p.add_argument("--triage", action="store_true",
                   help="print a severity-ranked triage view (most urgent first)")
    p.add_argument("--triage-top", type=int, default=0, metavar="N",
                   help="cap the triage view to N items per tier (0 = all)")
    p.add_argument("--suppress", metavar="BASELINE.json",
                   help="apply a suppression/baseline file; matched findings are "
                        "kept visible but do NOT trip the fail gate")
    p.add_argument("--write-baseline", metavar="OUT.json",
                   help="write a suppression file baselining every current "
                        "finding (by fingerprint), then exit 0")
    p.add_argument("--cache", metavar="CACHE.json",
                   help="incremental mode: reuse cached verdicts for unchanged "
                        "units (keyed by unit content + toolchain version); only "
                        "changed/new units are re-verified, and the cache is "
                        "updated in place")
    p.add_argument("--dashboard", metavar="OUT.html",
                   help="write a self-contained, offline migration-risk HTML "
                        "dashboard summarizing risk per divergence class")
    p.add_argument("--no-validate", action="store_true",
                   help="skip the shared-IR contract validation of the manifest "
                        "(by default every unit is validated and ill-formed "
                        "lowerings are rejected before verification)")
    p.add_argument("--verified-check", action="store_true",
                   help="for each source-UB-rooted DIVERGENT claim, build and "
                        "run the Lean/Lake verified checker over the raw "
                        "re-execution facts; failure is an operational error "
                        "(exit 2)")
    return p


def _print_text(reports: List[VerifyReport], units: List[Dict],
                summary: Dict, color: _Color, status_full: bool) -> None:
    out = sys.stdout
    out.write(color("1", "cross-lang-verify") +
              f"  ({len(reports)} unit(s); toolchain "
              f"{'available' if status_full else 'UNAVAILABLE — symbolic only'})\n")
    out.write("\n")
    for i, (unit, rep) in enumerate(zip(units, reports)):
        code, label = _VERDICT_STYLE[rep.verdict]
        out.write(f"  {color(code, label):<22}  "
                  f"{_unit_label(unit, i)}  "
                  f"{color('2', '[' + pair_of(rep) + ']')}\n")
        if rep.detail:
            out.write(f"      {color('2', rep.detail)}\n")
    out.write("\n")
    ov = summary["overall"]
    out.write(color("1", "Summary") + f" — {ov['total']} unit(s)\n")
    out.write(f"  decided   : {ov['decided']}  "
              f"({ov['decided_fraction']:.0%})  "
              f"[divergent={ov['divergent']}, "
              f"no_divergence_found={ov['no_divergence_found']}]\n")
    out.write(f"  abstained : {ov['abstained']}  "
              f"({ov['abstained_fraction']:.0%})  "
              f"[candidate={ov['candidate']}, not_covered={ov['not_covered']}]\n")
    out.write(f"  unknown   : {ov['unknown']}  ({ov['unknown_fraction']:.0%})\n")
    out.write("\n  by pair:\n")
    for pair, t in summary["by_pair"].items():
        out.write(f"    {pair:<16} decided={t['decided']} "
                  f"abstained={t['abstained']} unknown={t['unknown']}\n")
    out.write("\n  by class:\n")
    for cls, t in summary["by_class"].items():
        out.write(f"    {cls:<18} decided={t['decided']} "
                  f"abstained={t['abstained']} unknown={t['unknown']}\n")
    out.write("\n  " + color("2", summary["disclaimer"]) + "\n")


def _verified_observation(rep: VerifyReport) -> Optional[Dict[str, bool]]:
    """Extract the raw facts consumed by the Lean checker.

    This intentionally reads the three independent re-execution axes rather than
    the final ``confirmed`` flag, so the checker does not tautologically accept a
    verdict by being handed the verdict itself.
    """
    if rep.verdict is not VerifyVerdict.DIVERGENT:
        return None
    if rep.divergence is None or rep.divergence.reexec is None:
        return None
    rr = rep.divergence.reexec
    if not rr.available:
        return None
    return {
        "ub_reached": bool(rr.ub_reachable),
        "target_defined": bool(rr.rust_defined),
        "consequence": bool(rr.ub_consequential),
    }


def _run_verified_checks(reports: List[VerifyReport]) -> tuple:
    from . import mechanized_soundness as ms

    claims = []
    skipped_non_ub = 0
    for i, rep in enumerate(reports):
        if rep.verdict is VerifyVerdict.DIVERGENT:
            ce = rep.divergence.counterexample if rep.divergence else None
            if ce is None or ce.source_definedness != "undefined":
                skipped_non_ub += 1
                continue
            obs = _verified_observation(rep)
            if obs is None:
                return None, (
                    f"verified-check failed: divergent report {i} has no "
                    "available raw re-execution facts")
            claims.append((i, rep, obs))

    summary = {
        "enabled": True,
        "checked": 0,
        "accepted": 0,
        "kernel_theorem": "oracle_sound",
        "scope": "final source-UB positive-claim inference over trusted run facts",
        "checker_hash": None,
        "skipped_non_ub_rooted": skipped_non_ub,
    }
    if not claims:
        summary["status"] = "no_positive_claims"
        return summary, None

    build = ms.build_verified_checker()
    if not build.ok:
        return None, "verified-check build failed:\n" + ms.render_verified_checker_build(build)
    summary["checker_hash"] = build.source_hash

    for i, _rep, obs in claims:
        chk = ms.run_verified_checker(
            VerifyVerdict.DIVERGENT.value,
            obs["ub_reached"],
            obs["target_defined"],
            obs["consequence"],
            build=False,
        )
        if not chk.ok:
            return None, (
                f"verified-check rejected divergent report {i}: "
                f"exit={chk.exit_code} stdout={chk.stdout!r} stderr={chk.stderr!r}")
        summary["checked"] += 1
        summary["accepted"] += 1

    summary["status"] = "accepted"
    return summary, None


def run(argv: Optional[Sequence[str]] = None) -> int:
    args = create_parser().parse_args(argv)
    color = _Color(_want_color(args.color))
    fail_on = args.fail_on or ["divergent"]
    fail_verdicts = set()
    for tok in fail_on:
        fail_verdicts |= _FAIL_ON[tok]

    try:
        units = _load_units(args.units, validate=not args.no_validate)
    except IRValidationError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2

    status = toolchain_available()
    cache = None
    inc = None
    if args.cache:
        cache = VerificationCache.load(args.cache)
        inc = verify_incremental(units, cache, confirm=not args.no_confirm,
                                 status=status)
        reports: List[VerifyReport] = inc.reports
        cache.prune_to(units)
        cache.save(args.cache)
    else:
        reports = [
            verify_unit(u, confirm=not args.no_confirm, status=status)
            for u in units
        ]
    summary = aggregate_reports(reports)

    verified_summary = None
    if args.verified_check:
        verified_summary, err = _run_verified_checks(reports)
        if err is not None:
            sys.stderr.write(err + "\n")
            return 2

    # --write-baseline: capture every current finding and exit (adoption path).
    if args.write_baseline:
        baseline = build_baseline(reports)
        try:
            with open(args.write_baseline, "w", encoding="utf-8") as fh:
                json.dump(baseline, fh, indent=2, sort_keys=True)
                fh.write("\n")
        except OSError as exc:
            sys.stderr.write(f"error: cannot write baseline: {exc}\n")
            return 2
        n = len(baseline["suppressions"])
        sys.stderr.write(
            f"wrote {args.write_baseline}: baselined {n} finding(s)\n")
        return 0

    # --suppress: apply a baseline so known-accepted findings stop blocking CI.
    supp = None
    if args.suppress:
        try:
            rules = load_suppressions(args.suppress)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            sys.stderr.write(f"error: bad suppression file: {exc}\n")
            return 2
        supp = apply_suppressions(reports, rules)
        suppressed = {id(o.report) for o in supp.outcomes if o.suppressed}
        for rule in supp.expired_rules:
            sys.stderr.write(
                f"warning: suppression expired ({rule.expires}): {rule.reason}\n")
        for rule in supp.empty_rules:
            sys.stderr.write(
                f"warning: suppression matches EVERY finding (no constraints): "
                f"{rule.reason}\n")
        for rule in supp.unused_rules:
            sys.stderr.write(
                f"warning: suppression never matched: {rule.reason}\n")
    else:
        suppressed = set()

    if args.sarif:
        try:
            with open(args.sarif, "w", encoding="utf-8") as fh:
                json.dump(to_sarif(reports), fh, indent=2, sort_keys=True)
                fh.write("\n")
        except OSError as exc:
            sys.stderr.write(f"error: cannot write SARIF: {exc}\n")
            return 2

    if args.dashboard:
        try:
            with open(args.dashboard, "w", encoding="utf-8") as fh:
                fh.write(render_dashboard(reports))
        except OSError as exc:
            sys.stderr.write(f"error: cannot write dashboard: {exc}\n")
            return 2
        sys.stderr.write(f"wrote {args.dashboard}\n")

    if args.format == "json":
        json.dump({
            "summary": summary,
            "suppressed": len(suppressed),
            "cache": inc.to_dict() if inc is not None else None,
            "verified_check": verified_summary,
            "units": [
                {"label": _unit_label(u, i), "verdict": r.verdict.value,
                 "pair": pair_of(r), "detail": r.detail,
                 "suppressed": id(r) in suppressed}
                for i, (u, r) in enumerate(zip(units, reports))
            ],
        }, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        _print_text(reports, units, summary, color, status.full)
        if verified_summary is not None:
            sys.stdout.write(
                "\n  " + color(
                    "2",
                    "verified-check: "
                    f"{verified_summary['status']} "
                    f"({verified_summary['accepted']}/"
                    f"{verified_summary['checked']} source-UB claim(s); "
                    f"skipped_non_ub="
                    f"{verified_summary['skipped_non_ub_rooted']}; "
                    f"theorem={verified_summary['kernel_theorem']})",
                ) + "\n")
        if inc is not None:
            sys.stdout.write(
                "\n  " + color("2", f"cache: {inc.hits} hit(s), {inc.misses} "
                                    f"re-verified, {inc.stored} stored "
                                    f"({inc.hit_rate:.0%} reuse)") + "\n")
        if suppressed:
            sys.stdout.write(
                "\n  " + color("2", f"{len(suppressed)} finding(s) suppressed by "
                                    "baseline (not blocking)") + "\n")
        if args.triage:
            sys.stdout.write("\n" + render_triage(
                triage_reports(reports), color=color,
                max_per_tier=args.triage_top) + "\n")

    # Baseline-aware fail gate: a suppressed finding never blocks.
    return 1 if any(r.verdict in fail_verdicts and id(r) not in suppressed
                    for r in reports) else 0


def main(argv: Optional[Sequence[str]] = None) -> int:  # console-script entry
    return run(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
