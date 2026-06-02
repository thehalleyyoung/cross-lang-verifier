"""
Local, telemetry-free migration-risk dashboard (100_STEPS step 68).

After running the verifier over a project's units, a team wants a single,
shareable artifact that answers "how risky is this migration, and *where*?" —
without phoning home.  :func:`render_dashboard` turns a list of
:class:`~ub_oracle.verify.VerifyReport` into a self-contained HTML page:

* a headline risk posture (driven by the triage tiers),
* per-divergence-class risk rows (confirmed / candidate / clean counts and the
  catalogue severity), so the riskiest classes are obvious at a glance,
* a per-language-pair breakdown, and
* the full, severity-ranked finding list (reusing the triage ordering).

The page is **100% offline**: no external scripts, fonts, trackers or network
requests — all CSS is inlined, every number is computed locally, and the honesty
disclaimer ("NO-DIVERGENCE is not a proof of equivalence") is rendered in the
page itself.  It is deterministic given the same reports (stable ordering) so it
diffs cleanly and screenshots reproducibly.
"""

from __future__ import annotations

import datetime as _dt
import html
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from .catalogue import CATALOGUE
from .report import _class_of, pair_of
from .triage import Tier, severity_of_class, triage_reports
from .verify import VerifyReport, VerifyVerdict

_DISCLAIMER = (
    "This tool is SOUND FOR DIVERGENCE: a confirmed divergence is verified by "
    "really compiling and running the source and target. NO-DIVERGENCE means "
    "only the covered classes were checked — it is NOT a proof of equivalence.")


@dataclass
class ClassRisk:
    divergence_class: str
    severity: Optional[str]
    confirmed: int = 0
    candidate: int = 0
    clean: int = 0
    other: int = 0

    @property
    def total(self) -> int:
        return self.confirmed + self.candidate + self.clean + self.other

    @property
    def risk_score(self) -> int:
        """Crude orderable risk: confirmed dominate, candidates count less."""
        sev_w = {"critical": 100, "moderate": 30, "minor": 10}.get(
            self.severity or "", 50)
        return self.confirmed * sev_w + self.candidate * (sev_w // 5 + 1)

    def to_dict(self) -> Dict:
        return {
            "divergence_class": self.divergence_class,
            "severity": self.severity,
            "confirmed": self.confirmed,
            "candidate": self.candidate,
            "clean": self.clean,
            "other": self.other,
            "total": self.total,
            "risk_score": self.risk_score,
        }


def class_risks(reports: Iterable[VerifyReport]) -> List[ClassRisk]:
    """Aggregate per-divergence-class risk rows, ordered most-risky first."""
    rows: Dict[str, ClassRisk] = {}
    for r in reports:
        cls = (r.divergence.divergence_class
               if r.divergence is not None else _class_of(r))
        sev = severity_of_class(cls)
        row = rows.setdefault(cls, ClassRisk(cls, sev.value if sev else None))
        if r.verdict is VerifyVerdict.DIVERGENT:
            row.confirmed += 1
        elif r.verdict is VerifyVerdict.CANDIDATE:
            row.candidate += 1
        elif r.verdict is VerifyVerdict.NO_DIVERGENCE_FOUND:
            row.clean += 1
        else:
            row.other += 1
    return sorted(rows.values(),
                  key=lambda x: (-x.risk_score, x.divergence_class))


@dataclass
class DashboardData:
    total: int
    confirmed: int
    candidate: int
    clean: int
    unknown: int
    not_covered: int
    by_class: List[ClassRisk]
    by_pair: Dict[str, Dict[str, int]]
    posture: str

    def to_dict(self) -> Dict:
        return {
            "total": self.total,
            "confirmed": self.confirmed,
            "candidate": self.candidate,
            "clean": self.clean,
            "unknown": self.unknown,
            "not_covered": self.not_covered,
            "posture": self.posture,
            "by_class": [c.to_dict() for c in self.by_class],
            "by_pair": self.by_pair,
        }


def _posture(confirmed: int, candidate: int, total: int) -> str:
    if confirmed > 0:
        return "AT RISK"
    if candidate > 0:
        return "NEEDS REVIEW"
    if total > 0:
        return "NO DIVERGENCE FOUND"
    return "EMPTY"


def dashboard_data(reports: Iterable[VerifyReport]) -> DashboardData:
    reports = list(reports)
    counts = {v: 0 for v in VerifyVerdict}
    by_pair: Dict[str, Dict[str, int]] = {}
    for r in reports:
        counts[r.verdict] += 1
        p = by_pair.setdefault(pair_of(r), {"confirmed": 0, "candidate": 0,
                                            "clean": 0, "other": 0})
        if r.verdict is VerifyVerdict.DIVERGENT:
            p["confirmed"] += 1
        elif r.verdict is VerifyVerdict.CANDIDATE:
            p["candidate"] += 1
        elif r.verdict is VerifyVerdict.NO_DIVERGENCE_FOUND:
            p["clean"] += 1
        else:
            p["other"] += 1
    confirmed = counts[VerifyVerdict.DIVERGENT]
    candidate = counts[VerifyVerdict.CANDIDATE]
    return DashboardData(
        total=len(reports),
        confirmed=confirmed,
        candidate=candidate,
        clean=counts[VerifyVerdict.NO_DIVERGENCE_FOUND],
        unknown=counts[VerifyVerdict.UNKNOWN],
        not_covered=counts[VerifyVerdict.NOT_COVERED],
        by_class=class_risks(reports),
        by_pair={k: by_pair[k] for k in sorted(by_pair)},
        posture=_posture(confirmed, candidate, len(reports)),
    )


_CSS = """
:root{--bg:#0f1419;--card:#1a212b;--fg:#e6edf3;--muted:#8b949e;
--crit:#f85149;--mod:#d29922;--ok:#3fb950;--cand:#bb8009;--line:#30363d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
.wrap{max-width:960px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:22px;margin:0 0 4px}.sub{color:var(--muted);margin:0 0 22px;font-size:13px}
.posture{display:inline-block;padding:8px 16px;border-radius:8px;font-weight:700;
letter-spacing:.5px;margin:6px 0 24px}
.p-risk{background:rgba(248,81,73,.15);color:var(--crit);border:1px solid var(--crit)}
.p-review{background:rgba(210,153,34,.15);color:var(--mod);border:1px solid var(--mod)}
.p-ok{background:rgba(63,185,80,.15);color:var(--ok);border:1px solid var(--ok)}
.cards{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:26px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:14px 18px;min-width:120px;flex:1}
.card .n{font-size:26px;font-weight:700}.card .l{color:var(--muted);font-size:12px;text-transform:uppercase}
.n-crit{color:var(--crit)}.n-cand{color:var(--cand)}.n-ok{color:var(--ok)}
table{width:100%;border-collapse:collapse;margin:8px 0 26px;background:var(--card);
border:1px solid var(--line);border-radius:10px;overflow:hidden}
th,td{text-align:left;padding:9px 14px;border-bottom:1px solid var(--line);font-size:14px}
th{color:var(--muted);font-weight:600;text-transform:uppercase;font-size:11px;letter-spacing:.4px}
tr:last-child td{border-bottom:none}
.bar{height:8px;border-radius:4px;background:#222a35;overflow:hidden;min-width:90px}
.bar>span{display:block;height:100%;float:left}
.seg-crit{background:var(--crit)}.seg-cand{background:var(--cand)}.seg-ok{background:var(--ok)}
.pill{display:inline-block;padding:1px 8px;border-radius:10px;font-size:11px;font-weight:600}
.sev-critical{background:rgba(248,81,73,.18);color:var(--crit)}
.sev-moderate{background:rgba(210,153,34,.18);color:var(--mod)}
.sev-minor{background:rgba(139,148,158,.18);color:var(--muted)}
h2{font-size:15px;margin:26px 0 8px}
.disc{color:var(--muted);font-size:12px;border-left:3px solid var(--line);
padding:8px 14px;margin-top:30px;background:var(--card);border-radius:0 8px 8px 0}
.tier{font-weight:600}.t0{color:var(--crit)}.t1{color:#f0883e}.t2{color:var(--mod)}
.t3{color:var(--cand)}.t4{color:#a371f7}.t5{color:var(--muted)}.t6{color:var(--ok)}
"""


def _bar(row: ClassRisk) -> str:
    tot = row.total or 1
    def pct(n):
        return f"{100 * n / tot:.4f}%"
    return (f'<div class="bar">'
            f'<span class="seg-crit" style="width:{pct(row.confirmed)}"></span>'
            f'<span class="seg-cand" style="width:{pct(row.candidate)}"></span>'
            f'<span class="seg-ok" style="width:{pct(row.clean)}"></span>'
            f'</div>')


def _esc(s: str) -> str:
    return html.escape(str(s), quote=True)


def render_dashboard(reports: Iterable[VerifyReport], *,
                     title: str = "Cross-language migration risk",
                     generated_at: Optional[str] = None) -> str:
    """Render a self-contained, offline HTML dashboard string."""
    data = dashboard_data(reports)
    triage = triage_reports(reports)
    when = generated_at or _dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    posture_cls = {"AT RISK": "p-risk", "NEEDS REVIEW": "p-review"}.get(
        data.posture, "p-ok")

    cards = [
        ("total units", data.total, ""),
        ("confirmed divergences", data.confirmed, "n-crit"),
        ("candidates", data.candidate, "n-cand"),
        ("no-divergence", data.clean, "n-ok"),
        ("not covered", data.not_covered, ""),
        ("unknown", data.unknown, ""),
    ]
    cards_html = "".join(
        f'<div class="card"><div class="n {c}">{n}</div>'
        f'<div class="l">{_esc(l)}</div></div>'
        for (l, n, c) in cards)

    class_rows = "".join(
        f"<tr><td>{_esc(r.divergence_class)}</td>"
        f'<td>{_sev_pill(r.severity)}</td>'
        f'<td class="n-crit">{r.confirmed}</td>'
        f'<td class="n-cand">{r.candidate}</td>'
        f'<td class="n-ok">{r.clean}</td>'
        f"<td>{_bar(r)}</td></tr>"
        for r in data.by_class) or '<tr><td colspan="6">no units</td></tr>'

    pair_rows = "".join(
        f"<tr><td>{_esc(p)}</td>"
        f'<td class="n-crit">{t["confirmed"]}</td>'
        f'<td class="n-cand">{t["candidate"]}</td>'
        f'<td class="n-ok">{t["clean"]}</td>'
        f'<td>{t["other"]}</td></tr>'
        for p, t in data.by_pair.items()) or '<tr><td colspan="5">—</td></tr>'

    finding_rows = "".join(
        f'<tr><td class="tier t{int(it.tier)}">{_esc(it.tier.label)}</td>'
        f"<td>{_esc(it.unit_label)}</td>"
        f"<td>{_esc(it.divergence_class)}</td>"
        f'<td>{_sev_pill(it.severity)}</td>'
        f"<td>{_esc(it.pair)}</td></tr>"
        for it in triage.items
        if it.tier is not Tier.NO_DIVERGENCE) \
        or '<tr><td colspan="5">no findings</td></tr>'

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>{_CSS}</style></head>
<body><div class="wrap">
<h1>{_esc(title)}</h1>
<p class="sub">cross-lang-verifier · generated {_esc(when)} · offline / telemetry-free</p>
<div class="posture {posture_cls}">{_esc(data.posture)}</div>
<div class="cards">{cards_html}</div>

<h2>Risk by divergence class</h2>
<table><thead><tr><th>class</th><th>severity</th><th>confirmed</th>
<th>candidate</th><th>clean</th><th>mix</th></tr></thead>
<tbody>{class_rows}</tbody></table>

<h2>By language pair</h2>
<table><thead><tr><th>pair</th><th>confirmed</th><th>candidate</th>
<th>clean</th><th>other</th></tr></thead>
<tbody>{pair_rows}</tbody></table>

<h2>Findings (most urgent first)</h2>
<table><thead><tr><th>tier</th><th>unit</th><th>class</th>
<th>severity</th><th>pair</th></tr></thead>
<tbody>{finding_rows}</tbody></table>

<div class="disc">{_esc(_DISCLAIMER)}</div>
</div></body></html>
"""


def _sev_pill(severity: Optional[str]) -> str:
    if not severity:
        return '<span class="pill sev-minor">—</span>'
    return f'<span class="pill sev-{_esc(severity)}">{_esc(severity)}</span>'
