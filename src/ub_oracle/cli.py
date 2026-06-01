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
from .verify import VerifyReport, VerifyVerdict, verify_unit

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


def _load_units(path: str) -> List[Dict]:
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
            "  cross-lang-verify --units units.json --sarif out.sarif\n"
            "  cross-lang-verify --units units.json --format json --no-confirm\n"
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


def run(argv: Optional[Sequence[str]] = None) -> int:
    args = create_parser().parse_args(argv)
    color = _Color(_want_color(args.color))
    fail_on = args.fail_on or ["divergent"]
    fail_verdicts = set()
    for tok in fail_on:
        fail_verdicts |= _FAIL_ON[tok]

    try:
        units = _load_units(args.units)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2

    status = toolchain_available()
    reports: List[VerifyReport] = [
        verify_unit(u, confirm=not args.no_confirm, status=status) for u in units
    ]
    summary = aggregate_reports(reports)

    if args.sarif:
        try:
            with open(args.sarif, "w", encoding="utf-8") as fh:
                json.dump(to_sarif(reports), fh, indent=2, sort_keys=True)
                fh.write("\n")
        except OSError as exc:
            sys.stderr.write(f"error: cannot write SARIF: {exc}\n")
            return 2

    if args.format == "json":
        json.dump({
            "summary": summary,
            "units": [
                {"label": _unit_label(u, i), "verdict": r.verdict.value,
                 "pair": pair_of(r), "detail": r.detail}
                for i, (u, r) in enumerate(zip(units, reports))
            ],
        }, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        _print_text(reports, units, summary, color, status.full)

    return 1 if any(r.verdict in fail_verdicts for r in reports) else 0


def main(argv: Optional[Sequence[str]] = None) -> int:  # console-script entry
    return run(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
