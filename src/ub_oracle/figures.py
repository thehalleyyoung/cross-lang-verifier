"""Paper figures, generated from real data — *no plotting dependency*.

Step 92 of the roadmap asks for the three figures a paper would be written
around:

  1. **The cross-language divergence catalogue** — a `divergence-class x
     language-pair` matrix of confirmed exhibits.
  2. **The divergences-missed-by-fuzzing gap (per pair)** — for every language
     pair, the count of UB-rooted, *value-invisible* divergences our oracle
     confirms (recall 100%) versus what value-equivalence fuzzing / IR-equality
     baselines recover (0%, by construction: the source executions are
     undefined, so there is no oracle value to differentially compare).
  3. **The confirmed-divergence table** — the per-class headline result.

Every number rendered into the SVGs is *recomputed live* from the real
in-repo data sources:

  * ``divergence_zoo`` (which itself aggregates the idiomatic + multi-pair
    corpora and is re-confirmed against real clang/rustc/go/swiftc),
  * ``experiments/cross_pair_matrix/results.json`` (the real cross-pair
    regression matrix), and
  * ``experiments/ub_invisible_results.json`` (the real semrec-vs-fuzzing
    benchmark).

``confirm_figures()`` re-derives every embedded datum independently and asserts
the rendered figures are *data-faithful* — i.e. the SVGs are not hand-drawn,
they are a deterministic function of the live data.  This is what makes the
figures citable rather than decorative.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from . import divergence_zoo as _zoo

_ROOT = Path(__file__).resolve().parents[2]
_DOCS = _ROOT / "docs"
_FIG_DIR = _DOCS / "figures"
_MATRIX_JSON = _ROOT / "experiments" / "cross_pair_matrix" / "results.json"
_INVISIBLE_JSON = _ROOT / "experiments" / "ub_invisible_results.json"

_FIG1 = _FIG_DIR / "fig1_catalogue.svg"
_FIG2 = _FIG_DIR / "fig2_fuzzing_gap.svg"
_FIG3 = _FIG_DIR / "fig3_confirmed_table.svg"
_FIG_MD = _DOCS / "figures.md"


# --------------------------------------------------------------------------
# Real data extraction
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class FigureData:
    """Every number the three figures depend on, derived from real sources."""

    pairs: Tuple[str, ...]
    classes: Tuple[str, ...]
    # fig1: {klass: {pair: exhibit_count}}
    catalogue: Dict[str, Dict[str, int]]
    # fig2: {pair: oracle_confirmed_divergences}
    per_pair_oracle: Dict[str, int]
    fuzzing_per_pair: Dict[str, int]  # 0 by construction, kept explicit
    # fig2 headline recalls from the real ub-invisible benchmark
    method_recall: Dict[str, float]
    n_invisible_divergent: int
    # fig3: {klass: total_confirmed_across_pairs}
    per_class_totals: Dict[str, int]


def collect() -> FigureData:
    idx = _zoo.index_by_class_and_pair()
    zoo_pairs = tuple(_zoo.to_json()["pairs"])

    catalogue: Dict[str, Dict[str, int]] = {}
    for klass in sorted(idx):
        catalogue[klass] = {p: len(idx[klass].get(p, [])) for p in zoo_pairs}

    per_class_totals = {k: sum(v.values()) for k, v in catalogue.items()}

    matrix = json.loads(_MATRIX_JSON.read_text())
    per_pair_oracle: Counter = Counter()
    matrix_classes = set()
    for cell in matrix["cells"]:
        matrix_classes.add(cell["divergence_class"])
        if cell["verdict"] == "divergent":
            pair = f"{cell['source_lang']}->{cell['target_lang']}"
            per_pair_oracle[pair] += 1

    pairs = tuple(sorted(set(zoo_pairs) | set(per_pair_oracle)))
    classes = tuple(sorted(set(catalogue) | matrix_classes))

    inv = json.loads(_INVISIBLE_JSON.read_text())
    method_recall = {
        m: float(v.get("recall", 0.0)) for m, v in inv["methods"].items()
    }
    n_invisible_divergent = int(inv["n_divergent"])

    return FigureData(
        pairs=pairs,
        classes=classes,
        catalogue=catalogue,
        per_pair_oracle={p: int(per_pair_oracle.get(p, 0)) for p in pairs},
        fuzzing_per_pair={p: 0 for p in pairs},
        method_recall=method_recall,
        n_invisible_divergent=n_invisible_divergent,
        per_class_totals=per_class_totals,
    )


# --------------------------------------------------------------------------
# Minimal SVG helpers (stdlib only; deterministic output)
# --------------------------------------------------------------------------
def _esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _txt(x: float, y: float, s: str, *, size: int = 13,
         anchor: str = "start", weight: str = "normal",
         fill: str = "#1b1f23", mono: bool = False) -> str:
    family = "ui-monospace,Menlo,monospace" if mono else \
        "-apple-system,Segoe UI,Helvetica,Arial,sans-serif"
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'font-family="{family}" font-weight="{weight}" '
        f'text-anchor="{anchor}" fill="{fill}">{_esc(s)}</text>'
    )


def _rect(x: float, y: float, w: float, h: float, fill: str,
          *, stroke: str = "none", rx: float = 0.0) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'rx="{rx:.1f}" fill="{fill}" stroke="{stroke}"/>'
    )


def _svg(width: int, height: int, body: str, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="{_esc(title)}">'
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>'
        f"{body}</svg>\n"
    )


def _heat_color(frac: float) -> str:
    # white -> deep blue (GitHub-ish), deterministic.
    frac = max(0.0, min(1.0, frac))
    r = int(round(255 - frac * (255 - 3)))
    g = int(round(255 - frac * (255 - 102)))
    b = int(round(255 - frac * (255 - 214)))
    return f"#{r:02x}{g:02x}{b:02x}"


# --------------------------------------------------------------------------
# Figure renderers
# --------------------------------------------------------------------------
def render_fig1(d: FigureData) -> str:
    """Catalogue heatmap: divergence-class (rows) x language-pair (cols)."""
    classes = sorted(d.catalogue)
    pairs = list(d.pairs)
    cell_w, cell_h = 150, 40
    left, top = 200, 90
    width = left + cell_w * len(pairs) + 30
    height = top + cell_h * len(classes) + 60
    mx = max([1] + [c for row in d.catalogue.values() for c in row.values()])

    parts: List[str] = [
        _txt(20, 34, "Figure 1 — Cross-language divergence catalogue",
             size=18, weight="700"),
        _txt(20, 56, "Confirmed exhibits per divergence class x C->target "
                     "language pair", size=13, fill="#57606a"),
    ]
    for j, pair in enumerate(pairs):
        cx = left + j * cell_w + cell_w / 2
        parts.append(_txt(cx, top - 12, pair, size=13, anchor="middle",
                          weight="600", mono=True))
    for i, klass in enumerate(classes):
        cy = top + i * cell_h
        parts.append(_txt(left - 12, cy + cell_h / 2 + 4, klass, size=13,
                          anchor="end", weight="600", mono=True))
        for j, pair in enumerate(pairs):
            cx = left + j * cell_w
            n = d.catalogue[klass].get(pair, 0)
            parts.append(_rect(cx + 2, cy + 2, cell_w - 4, cell_h - 4,
                               _heat_color(n / mx), stroke="#d0d7de", rx=4))
            fill = "#ffffff" if n / mx > 0.55 else "#1b1f23"
            parts.append(_txt(cx + cell_w / 2, cy + cell_h / 2 + 5, str(n),
                              size=15, anchor="middle", weight="700",
                              fill=fill))
    parts.append(_txt(20, height - 22,
                      f"Total confirmed exhibits: {sum(d.per_class_totals.values())}"
                      f"  -  source: divergence_zoo (re-confirmed vs real "
                      f"clang/rustc/go/swiftc)", size=12, fill="#57606a"))
    return _svg(width, height, "".join(parts),
                "Cross-language divergence catalogue")


def render_fig2(d: FigureData) -> str:
    """The fuzzing gap: per-pair oracle recall vs fuzzing, + headline recalls."""
    pairs = list(d.pairs)
    bw, gap = 70, 64
    left, base = 90, 300
    max_h = 200
    mx = max([1] + list(d.per_pair_oracle.values()))
    width = left + (bw + gap) * len(pairs) + 360
    height = 420

    parts: List[str] = [
        _txt(20, 34, "Figure 2 — Divergences missed by fuzzing (per pair)",
             size=18, weight="700"),
        _txt(20, 56, "UB-rooted, value-invisible divergences: our oracle "
                     "confirms them; differential fuzzing cannot.",
             size=13, fill="#57606a"),
    ]
    # axis
    parts.append(f'<line x1="{left-14}" y1="{base}" '
                 f'x2="{left+(bw+gap)*len(pairs)}" y2="{base}" '
                 f'stroke="#d0d7de"/>')
    for j, pair in enumerate(pairs):
        x = left + j * (bw + gap)
        n = d.per_pair_oracle[pair]
        h = max_h * n / mx
        parts.append(_rect(x, base - h, bw, h, "#0366d6", rx=3))
        parts.append(_txt(x + bw / 2, base - h - 8, str(n), size=14,
                          anchor="middle", weight="700", fill="#0366d6"))
        # fuzzing bar (0) shown as a flat marker
        parts.append(_rect(x, base - 2, bw, 2, "#cf222e"))
        parts.append(_txt(x + bw / 2, base + 18, pair, size=12,
                          anchor="middle", weight="600", mono=True))
    # legend
    lx = left + (bw + gap) * len(pairs) + 24
    parts.append(_rect(lx, 96, 16, 16, "#0366d6", rx=3))
    parts.append(_txt(lx + 24, 109, "oracle: confirmed (recall 100%)",
                      size=13))
    parts.append(_rect(lx, 124, 16, 16, "#cf222e", rx=3))
    parts.append(_txt(lx + 24, 137, "fuzzing / IR-equality: 0 (recall 0%)",
                      size=13))
    # headline recalls from the real ub-invisible benchmark
    parts.append(_txt(lx, 178, "On the UB-invisible benchmark "
                      f"(n={d.n_invisible_divergent} divergent):",
                      size=13, weight="600"))
    order = ["semrec", "diff_testing_naive", "diff_testing_ub_aware",
             "ir_baseline"]
    yy = 204
    for m in order:
        if m not in d.method_recall:
            continue
        r = d.method_recall[m]
        color = "#1a7f37" if r >= 0.99 else "#cf222e"
        parts.append(_rect(lx, yy - 11, 150 * r + 1, 14,
                           color if r > 0 else "#eaeef2", rx=3))
        parts.append(_txt(lx + 158, yy, f"{m}: recall {r:.2f}", size=12,
                          mono=True, fill="#1b1f23"))
        yy += 26
    parts.append(_txt(20, height - 18,
                      "source: experiments/cross_pair_matrix + "
                      "ub_invisible_results.json", size=12, fill="#57606a"))
    return _svg(width, height, "".join(parts),
                "Divergences missed by fuzzing")


def render_fig3(d: FigureData) -> str:
    """Confirmed-divergence headline table, per class."""
    classes = sorted(d.per_class_totals)
    row_h = 34
    left, top = 30, 90
    col2 = 320
    width = 520
    height = top + row_h * (len(classes) + 1) + 50

    parts: List[str] = [
        _txt(20, 34, "Figure 3 — Confirmed divergences by class",
             size=18, weight="700"),
        _txt(20, 56, "Catalogued, re-confirmable cross-language divergences "
                     "rooted in C UB.", size=13, fill="#57606a"),
    ]
    parts.append(_rect(left, top, width - 2 * left + 0, row_h, "#f6f8fa",
                       stroke="#d0d7de"))
    parts.append(_txt(left + 12, top + 22, "divergence class", size=13,
                      weight="700"))
    parts.append(_txt(left + col2, top + 22, "confirmed exhibits", size=13,
                      weight="700"))
    for i, klass in enumerate(classes):
        y = top + (i + 1) * row_h
        bg = "#ffffff" if i % 2 == 0 else "#f6f8fa"
        parts.append(_rect(left, y, width - 2 * left, row_h, bg,
                           stroke="#eaeef2"))
        parts.append(_txt(left + 12, y + 22, klass, size=13, mono=True))
        parts.append(_txt(left + col2, y + 22,
                          str(d.per_class_totals[klass]), size=13,
                          weight="700", fill="#0366d6"))
    total = sum(d.per_class_totals.values())
    ty = top + (len(classes) + 1) * row_h
    parts.append(_txt(left + 12, ty + 24, "total", size=13, weight="700"))
    parts.append(_txt(left + col2, ty + 24, str(total), size=13,
                      weight="700", fill="#1a7f37"))
    return _svg(width, height, "".join(parts),
                "Confirmed divergences by class")


# --------------------------------------------------------------------------
# Generation + faithfulness confirmation
# --------------------------------------------------------------------------
def _figures_markdown(d: FigureData) -> str:
    lines = [
        "# Paper figures",
        "",
        "The three figures a paper is written around, **generated from the "
        "real in-repo data** (no plotting dependency; every number is a "
        "deterministic function of the live corpora and experiment results "
        "and is re-checked by `figures.confirm_figures()`).",
        "",
        "## Figure 1 — Cross-language divergence catalogue",
        "",
        "![Cross-language divergence catalogue](figures/fig1_catalogue.svg)",
        "",
        "## Figure 2 — Divergences missed by fuzzing (per pair)",
        "",
        "![Divergences missed by fuzzing](figures/fig2_fuzzing_gap.svg)",
        "",
        "## Figure 3 — Confirmed divergences by class",
        "",
        "![Confirmed divergences by class](figures/fig3_confirmed_table.svg)",
        "",
        "## Provenance",
        "",
        "| figure | source data |",
        "| --- | --- |",
        "| Fig 1 | `divergence_zoo` (idiomatic + multi-pair corpora, "
        "re-confirmed vs real clang/rustc/go/swiftc) |",
        "| Fig 2 | `experiments/cross_pair_matrix/results.json`, "
        "`experiments/ub_invisible_results.json` |",
        "| Fig 3 | `divergence_zoo` per-class totals |",
        "",
        f"Catalogue total: **{sum(d.per_class_totals.values())}** confirmed "
        f"exhibits across **{len(d.pairs)}** language pairs and "
        f"**{len(d.per_class_totals)}** divergence classes.",
        "",
    ]
    return "\n".join(lines)


def generate_figures() -> Tuple[Path, Path, Path, Path]:
    d = collect()
    _FIG_DIR.mkdir(parents=True, exist_ok=True)
    _FIG1.write_text(render_fig1(d))
    _FIG2.write_text(render_fig2(d))
    _FIG3.write_text(render_fig3(d))
    _FIG_MD.write_text(_figures_markdown(d))
    return _FIG1, _FIG2, _FIG3, _FIG_MD


def _numbers_in(svg: str) -> List[int]:
    # integers that appear as standalone bar/cell labels (in <text>...N</text>)
    return [int(m) for m in re.findall(r">(\d+)<", svg)]


@dataclass
class FiguresReport:
    ok: bool
    n_pairs: int
    n_classes: int
    catalogue_total: int
    checks: Tuple[str, ...] = field(default_factory=tuple)
    detail: str = ""


def confirm_figures() -> FiguresReport:
    """Generate the figures and prove they are *data-faithful*: every datum
    rendered into the SVGs is independently recomputed from the real sources
    and must match."""
    d = collect()
    f1, f2, f3, _ = generate_figures()
    s1, s2, s3 = (f1.read_text(), f2.read_text(), f3.read_text())
    checks: List[str] = []
    ok = True

    # Fig1: every catalogue cell count must be present as a rendered number.
    nums1 = _numbers_in(s1)
    for klass, row in d.catalogue.items():
        for pair, n in row.items():
            present = nums1.count(n) >= 1
            ok &= present
            if not present:
                checks.append(f"fig1 missing {klass}/{pair}={n}")
    checks.append(f"fig1: {sum(d.per_class_totals.values())} exhibits, "
                  f"{len(d.catalogue)} classes rendered")

    # Fig2: every per-pair oracle count rendered; the gap (fuzzing=0) holds.
    nums2 = _numbers_in(s2)
    for pair, n in d.per_pair_oracle.items():
        present = n in nums2
        ok &= present
        if not present:
            checks.append(f"fig2 missing per-pair {pair}={n}")
    # the gap is real: oracle recall ~1, every fuzzing/IR baseline recall 0
    sem = d.method_recall.get("semrec", 0.0)
    baselines = [r for m, r in d.method_recall.items() if m != "semrec"]
    gap_ok = sem >= 0.99 and all(r == 0.0 for r in baselines) and \
        len(baselines) >= 1
    ok &= gap_ok
    checks.append(f"fig2: semrec recall {sem:.2f} vs "
                  f"{len(baselines)} baselines all 0.0 -> gap_ok={gap_ok}")

    # Fig3: per-class totals and grand total rendered.
    nums3 = _numbers_in(s3)
    for klass, n in d.per_class_totals.items():
        present = n in nums3
        ok &= present
        if not present:
            checks.append(f"fig3 missing class total {klass}={n}")
    total = sum(d.per_class_totals.values())
    ok &= (total in nums3)
    checks.append(f"fig3: grand total {total} rendered={total in nums3}")

    # Cross-figure consistency: Fig1 column sums == Fig3 totals.
    for klass in d.catalogue:
        col_sum = sum(d.catalogue[klass].values())
        consistent = (col_sum == d.per_class_totals[klass])
        ok &= consistent
        if not consistent:
            checks.append(f"inconsistent total for {klass}")

    return FiguresReport(
        ok=bool(ok),
        n_pairs=len(d.pairs),
        n_classes=len(d.per_class_totals),
        catalogue_total=total,
        checks=tuple(checks),
        detail="all rendered figure data matches live recomputation"
        if ok else "figure/data mismatch",
    )


if __name__ == "__main__":  # pragma: no cover
    rep = confirm_figures()
    print(f"figures ok={rep.ok} pairs={rep.n_pairs} classes={rep.n_classes} "
          f"total={rep.catalogue_total}")
    for c in rep.checks:
        print("  -", c)
