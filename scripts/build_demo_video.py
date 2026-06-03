#!/usr/bin/env python3
"""Build the README-linked three-minute demo video from live evidence.

The video is intentionally a derived artifact: before rendering any frame, this
script compiles and runs a checked-in c2rust-generated Rust function and its C
source under real toolchains.  The primary scene is a CWE-369-class division by
zero in the ``nginx-rate`` c2rust corpus unit; the teaser scene confirms the same
class for C->Go through the existing CVE-class corpus.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ub_oracle import c2rust_corpus, cve_corpus
from src.ub_oracle.reexec import ReexecHarness, ReexecResult, RunOutcome, toolchain_available
from src.ub_oracle.verify import verify_unit


DEFAULT_OUTPUT = REPO_ROOT / "docs" / "assets" / "demo_video.mp4"
DEFAULT_POSTER = REPO_ROOT / "docs" / "assets" / "demo_video_poster.png"
DEFAULT_EVIDENCE = REPO_ROOT / "docs" / "assets" / "demo_video_evidence.json"
PRIMARY_ITEM_ID = "nginx-rate"
PRIMARY_INPUTS = ["7", "0"]
VIDEO_SCHEMA = "cross-lang-verifier-demo-video/v1"
DISPLAY_COMMAND = "python3 scripts/build_demo_video.py"

StyledLine = Tuple[str, str]


def sanitize_text(text: str, *, limit: Optional[int] = None) -> str:
    """Normalize environment-specific text while preserving diagnostics."""
    text = re.sub(r"\x1b\[[0-9;]*m", "", text or "")
    text = text.replace(str(REPO_ROOT), "<repo>")
    text = re.sub(r"/(?:private/)?var/[^\s:]*", "<tmp>", text)
    text = re.sub(r"/tmp/[^\s:]*", "<tmp>", text)
    text = re.sub(r"thread 'main' \(\d+\)", "thread 'main' (<tid>)", text)
    text = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", text)
    text = re.sub(r"\s+\n", "\n", text).strip()
    if limit is not None and len(text) > limit:
        return text[: max(0, limit - 1)].rstrip() + "..."
    return text


def _sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _primary_item() -> c2rust_corpus.C2RustItem:
    for item in c2rust_corpus.CORPUS:
        if item.item_id == PRIMARY_ITEM_ID:
            return item
    raise RuntimeError(f"missing c2rust corpus item: {PRIMARY_ITEM_ID}")


def _c_program_for_item(item: c2rust_corpus.C2RustItem) -> str:
    c_src = item.c_path.read_text(encoding="utf-8")
    return (
        "#include <stdio.h>\n"
        "#include <stdlib.h>\n\n"
        f"{c_src.rstrip()}\n\n"
        "int main(int argc, char **argv) {\n"
        "  if (argc != 3) return 64;\n"
        "  int bytes = atoi(argv[1]);\n"
        "  int seconds = atoi(argv[2]);\n"
        "  printf(\"%d\\n\", nginx_rate(bytes, seconds));\n"
        "  return 0;\n"
        "}\n"
    )


def _rust_program_for_item(item: c2rust_corpus.C2RustItem) -> str:
    rust_src = item.rust_path.read_text(encoding="utf-8")
    return (
        f"{rust_src.rstrip()}\n\n"
        "extern crate core;\n\n"
        "fn main() {\n"
        "    let mut args = std::env::args().skip(1);\n"
        "    let bytes: i32 = args.next().unwrap().parse().unwrap();\n"
        "    let seconds: i32 = args.next().unwrap().parse().unwrap();\n"
        "    let out = unsafe { nginx_rate(bytes, seconds) };\n"
        "    println!(\"{}\", out);\n"
        "}\n"
    )


def _outcome_dict(outcome: Optional[RunOutcome]) -> Dict[str, object]:
    if outcome is None:
        return {"available": False}
    return {
        "returncode": outcome.returncode,
        "stdout": sanitize_text(outcome.stdout, limit=240),
        "stderr": sanitize_text(outcome.stderr, limit=520),
        "timed_out": outcome.timed_out,
        "ub_category": outcome.ub_category,
    }


def _result_dict(result: ReexecResult, *, target_lang: str) -> Dict[str, object]:
    reason = sanitize_text(result.reason, limit=520)
    if target_lang != "rust":
        reason = reason.replace("Rust defined", f"{target_lang} defined")
    return {
        "available": result.available,
        "confirmed": result.confirmed,
        "mode": result.mode,
        "divergence_class": result.divergence_class,
        "inputs": dict(result.inputs),
        "ub_reachable": result.ub_reachable,
        "target_defined": result.rust_defined,
        "ub_consequential": result.ub_consequential,
        "target_lang": target_lang,
        "reason": reason,
        "c_runs": {name: _outcome_dict(run) for name, run in result.c_runs.items()},
        "target_run": _outcome_dict(result.rust_run),
    }


def _go_teaser_case() -> cve_corpus.CveCase:
    for case in cve_corpus.CORPUS:
        if case.case_id == "div-by-zero":
            return case
    raise RuntimeError("missing CVE-class div-by-zero case")


def _rust_ffi_abort_is_defined_panic(outcome: RunOutcome) -> bool:
    """Recognize rustc's deterministic abort for panics inside extern "C" code."""
    stderr = outcome.stderr
    return (
        outcome.returncode in (-6, 134)
        and "attempt to divide by zero" in stderr
        and "panic in a function that cannot unwind" in stderr
    )


def confirm_primary_c2rust_witness(status=None) -> ReexecResult:
    """Confirm the c2rust FFI witness against real C and Rust compilers.

    The c2rust artifact is an ``extern "C"`` function.  On current Rust, a
    divide-by-zero panic inside such a function aborts because the ABI cannot
    unwind.  That is still a deterministic Rust runtime outcome, so the demo
    records this FFI-specific proof shape instead of broadening the global Rust
    target-semantics pack.
    """
    item = _primary_item()
    st = status or toolchain_available()
    result = ReexecResult(
        available=st.full_for("rust"),
        divergence_class="div_by_zero",
        mode="c2rust_ffi_trap_vs_rust_abort",
        inputs={f"arg{i}": v for i, v in enumerate(PRIMARY_INPUTS)},
    )
    if not st.full_for("rust"):
        result.reason = "toolchain unavailable: " + ", ".join(
            ReexecHarness._missing_for(st, "rust"))
        return result

    harness = ReexecHarness(st)
    with tempfile.TemporaryDirectory() as workdir:
        san = harness._compile_c(
            _c_program_for_item(item),
            ["-O1", "-fsanitize=undefined", "-fno-sanitize-recover=all"],
            workdir,
            "c_san",
        )
        o0 = harness._compile_c(_c_program_for_item(item), ["-O0"], workdir, "c_o0")
        rust = harness._compile_target(_rust_program_for_item(item), "rust", workdir, "tgt")
        if not all((san, o0, rust)):
            result.available = False
            result.reason = "compilation failed (san=%s o0=%s tgt=%s)" % (
                bool(san), bool(o0), bool(rust))
            return result

        result.c_runs["san"] = harness._run([san, *PRIMARY_INPUTS])
        result.c_runs["O0"] = harness._run([o0, *PRIMARY_INPUTS])
        rust_a = harness._run_target(rust, "rust", PRIMARY_INPUTS)
        rust_b = harness._run_target(rust, "rust", PRIMARY_INPUTS)
        result.rust_run = rust_a

    rust_deterministic = (
        rust_a.returncode == rust_b.returncode
        and rust_a.stdout == rust_b.stdout
        and bool(rust_a.stderr) == bool(rust_b.stderr)
    )
    result.ub_reachable = result.c_runs["san"].ub_trapped
    result.rust_defined = rust_deterministic and (
        rust_a.target_outcome_defined("rust") or _rust_ffi_abort_is_defined_panic(rust_a)
    )
    result.ub_consequential = result.ub_reachable and result.rust_defined
    result.confirmed = result.ub_reachable and result.rust_defined
    if result.confirmed:
        result.reason = (
            "UB reachable in C (UBSan trapped: "
            f"{result.c_runs['san'].ub_category or 'division by zero'}); "
            f"c2rust Rust runtime outcome deterministic (rc={rust_a.returncode})"
        )
    else:
        result.reason = (
            f"not confirmed: ub_reachable={result.ub_reachable}, "
            f"rust_runtime_deterministic={rust_deterministic}, "
            f"rust_runtime_defined={result.rust_defined}"
        )
    return result


def _toolchain_summary() -> Dict[str, object]:
    status = toolchain_available()
    return {
        "c_compiler": bool(status.cc),
        "ubsan": status.ubsan,
        "rustc": status.target_available("rust"),
        "go": status.target_available("go"),
    }


def build_evidence(*, require_go_teaser: bool = True) -> Dict[str, object]:
    """Compile and run the primary/teaser witnesses, returning sanitized evidence."""
    item = _primary_item()
    status = toolchain_available()
    if not status.full_for("rust"):
        missing = ReexecHarness._missing_for(status, "rust")
        raise RuntimeError("primary C->Rust demo toolchain unavailable: " + ", ".join(missing))

    harness = ReexecHarness(status)
    primary_result = confirm_primary_c2rust_witness(status)
    if not primary_result.confirmed:
        raise RuntimeError("primary c2rust witness did not confirm: " + primary_result.reason)

    record = c2rust_corpus.case_record(item)
    symbolic = verify_unit(dict(item.unit), confirm=False, status=status)
    if symbolic.verdict.value != item.expected_symbolic_verdict:
        raise RuntimeError(
            f"symbolic c2rust verdict drifted: {symbolic.verdict.value} "
            f"!= {item.expected_symbolic_verdict}"
        )

    teaser_case = _go_teaser_case()
    teaser_src = teaser_case.target_for("go")
    if teaser_src is None:
        raise RuntimeError("CVE-class div-by-zero case has no Go target")
    if status.full_for("go"):
        teaser_result = harness.confirm_trap_vs_defined(
            teaser_case.c_src,
            teaser_src,
            list(teaser_case.inputs),
            divergence_class=teaser_case.divergence_class,
            target_lang="go",
        )
        if not teaser_result.confirmed:
            raise RuntimeError("C->Go teaser witness did not confirm: " + teaser_result.reason)
        teaser = {
            "available": True,
            "pair": "c->go",
            "case_id": teaser_case.case_id,
            "cwe": teaser_case.cwe,
            "title": teaser_case.title,
            "confirmation": _result_dict(teaser_result, target_lang="go"),
        }
    elif require_go_teaser:
        missing = ReexecHarness._missing_for(status, "go")
        raise RuntimeError("C->Go teaser toolchain unavailable: " + ", ".join(missing))
    else:
        teaser = {
            "available": False,
            "pair": "c->go",
            "case_id": teaser_case.case_id,
            "cwe": teaser_case.cwe,
            "title": teaser_case.title,
            "reason": "toolchain unavailable",
        }

    c_src = item.c_path.read_text(encoding="utf-8")
    rust_src = item.rust_path.read_text(encoding="utf-8")
    return {
        "schema": VIDEO_SCHEMA,
        "title": "Three-minute c2rust CWE-class divergence demo",
        "duration_seconds": 180,
        "generated_by": "scripts/build_demo_video.py",
        "toolchain": _toolchain_summary(),
        "primary": {
            "item_id": item.item_id,
            "source_library": item.source_library,
            "source_function": item.source_function,
            "provenance": item.provenance,
            "cwe": "CWE-369",
            "cve_claim": "CVE-class weakness only; no assigned nginx CVE is claimed",
            "translator": c2rust_corpus.TRANSLATOR,
            "translator_version": c2rust_corpus.TRANSLATOR_VERSION,
            "c_file": str(item.c_path.relative_to(REPO_ROOT)),
            "rust_file": str(item.rust_path.relative_to(REPO_ROOT)),
            "source_sha256": _sha256_text(c_src),
            "rust_sha256": _sha256_text(rust_src),
            "c2rust_hallmarks": record["c2rust_hallmarks"],
            "symbolic_verdict": symbolic.verdict.value,
            "symbolic_detail": sanitize_text(symbolic.detail, limit=320),
            "confirmation": _result_dict(primary_result, target_lang="rust"),
        },
        "teaser": teaser,
    }


def build_story(evidence: Dict[str, object]) -> List[List[StyledLine]]:
    """Convert evidence into terminal-style scenes."""
    primary = evidence["primary"]
    confirmation = primary["confirmation"]
    san = confirmation["c_runs"]["san"]
    target = confirmation["target_run"]
    teaser = evidence["teaser"]
    teaser_conf = teaser.get("confirmation", {}) if teaser.get("available") else {}
    teaser_target = teaser_conf.get("target_run", {})

    scenes: List[List[StyledLine]] = [
        [
            ("accent", "cross-lang-verifier"),
            ("text", "Three-minute demo generated from real compiler runs."),
            ("dim", "Evidence is captured from clang/UBSan, rustc, and go."),
            ("text", ""),
            ("prompt", "Primary witness"),
            ("text", "CWE-369-class division-by-zero in checked-in c2rust output"),
            ("text", f"{primary['source_library']} / {primary['source_function']}"),
            ("dim", "CVE-class weakness only; this does not claim an assigned nginx CVE."),
        ],
        [
            ("accent", "machine-translated artifact"),
            ("text", f"C source : {primary['c_file']}"),
            ("text", f"Rust out : {primary['rust_file']}"),
            ("text", f"Translator: {primary['translator']} {primary['translator_version']}"),
            ("text", f"c2rust ABI hallmarks: {primary['c2rust_hallmarks']}"),
            ("text", f"Symbolic verdict: {primary['symbolic_verdict']}"),
            ("dim", str(primary["symbolic_detail"])),
        ],
        [
            ("accent", "ground-truth replay"),
            ("command", "$ clang -O1 -fsanitize=undefined ... nginx_rate 7 0"),
            ("danger", f"UBSan category: {san.get('ub_category') or 'division by zero'}"),
            ("dim", str(san.get("stderr", ""))[:160]),
            ("text", ""),
            ("command", "$ rustc -O ... nginx_rate.rs && ./nginx_rate 7 0"),
            ("ok", f"Rust runtime outcome: rc={target.get('returncode')}"),
            ("dim", str(target.get("stderr", ""))[:160] or "deterministic target result"),
        ],
        [
            ("accent", "verdict"),
            ("danger", "DIVERGENT [div_by_zero]"),
            ("text", "C executes undefined behavior on the witness input."),
            ("text", "The c2rust-generated Rust target has a deterministic Rust runtime panic/abort."),
            ("ok", "The positive claim is made only after real compiler replay confirms it."),
            ("dim", str(confirmation["reason"])[:220]),
        ],
        [
            ("accent", "second-pair teaser"),
            ("text", f"{teaser.get('cwe', 'CWE')} {teaser.get('title', '')}"),
            ("text", "C -> Go uses the same replay discipline as the primary witness."),
            ("ok" if teaser.get("available") else "danger",
             "confirmed against go" if teaser.get("available") else "Go toolchain unavailable"),
            ("text", f"Go target outcome: rc={teaser_target.get('returncode', 'n/a')}"),
            ("dim", str(teaser_target.get("stderr", ""))[:180] or str(teaser.get("reason", ""))),
        ],
        [
            ("accent", "reproduce the media artifact"),
            ("command", "$ " + DISPLAY_COMMAND),
            ("text", "1. confirms the c2rust CWE-class witness"),
            ("text", "2. confirms the C->Go teaser witness"),
            ("text", "3. renders this MP4 and README poster from that evidence"),
            ("ok", "If replay fails, rendering fails."),
        ],
    ]
    return scenes


def _styled(style: str, text: str = "") -> StyledLine:
    return (style, text)


def _load_font(image_font_module, size: int):
    candidates = [
        "/System/Library/Fonts/Menlo.ttc",
        "/Library/Fonts/Menlo.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return image_font_module.truetype(candidate, size)
    return image_font_module.load_default()


def render_frame(image_module, draw_module, font, lines: Sequence[StyledLine],
                 size: Tuple[int, int] = (1280, 720)):
    width, height = size
    image = image_module.new("RGB", size, "#07111f")
    draw = draw_module.Draw(image)
    draw.rounded_rectangle((24, 24, width - 24, height - 24), radius=22, fill="#0f172a")
    draw.rounded_rectangle((24, 24, width - 24, height - 24), radius=22, outline="#334155", width=2)
    for i, color in enumerate(("#ef4444", "#f59e0b", "#22c55e")):
        draw.ellipse((52 + i * 30, 52, 72 + i * 30, 72), fill=color)
    draw.text((158, 47), "SemRec: c2rust CWE-class divergence, confirmed live", fill="#93c5fd", font=font)

    palette = {
        "accent": "#60a5fa",
        "command": "#fbbf24",
        "danger": "#fb7185",
        "dim": "#94a3b8",
        "ok": "#34d399",
        "prompt": "#a7f3d0",
        "text": "#e5e7eb",
    }
    y = 104
    for style, text in lines:
        if y > height - 60:
            break
        draw.text((58, y), text, fill=palette.get(style, palette["text"]), font=font)
        y += 38
    return image


def render_poster(evidence: Dict[str, object], poster_path: Path) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:  # pragma: no cover - optional local tooling
        raise RuntimeError("Pillow is required to render the demo poster") from exc

    poster_path.parent.mkdir(parents=True, exist_ok=True)
    font = _load_font(ImageFont, 26)
    poster = render_frame(Image, ImageDraw, font, build_story(evidence)[0])
    poster.save(poster_path)


def _durations(total_seconds: float, n_frames: int) -> List[float]:
    if n_frames <= 0:
        return []
    # The concat demuxer repeats the last image once so the final duration is
    # honored; account for that extra still frame to keep the MP4 near the
    # requested wall-clock length.
    each = float(total_seconds) / (n_frames + 1)
    return [max(0.2, each) for _ in range(n_frames)]


def render_video(
    evidence: Dict[str, object],
    output_path: Path,
    poster_path: Path,
    *,
    total_seconds: float = 180,
) -> None:
    """Render a silent MP4 and poster from evidence-derived frames."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to render docs/assets/demo_video.mp4")
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:  # pragma: no cover - optional local tooling
        raise RuntimeError("Pillow is required to render docs/assets/demo_video.mp4") from exc

    frames = build_story(evidence)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    poster_path.parent.mkdir(parents=True, exist_ok=True)
    font = _load_font(ImageFont, 26)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        frame_paths: List[Path] = []
        for idx, frame in enumerate(frames):
            img = render_frame(Image, ImageDraw, font, frame)
            fpath = tmp_path / f"frame_{idx:02d}.png"
            img.save(fpath)
            frame_paths.append(fpath)
            if idx == 0:
                img.save(poster_path)

        concat = tmp_path / "frames.txt"
        durations = _durations(total_seconds, len(frame_paths))
        lines: List[str] = []
        for fpath, duration in zip(frame_paths, durations):
            lines.append(f"file '{fpath.as_posix()}'")
            lines.append(f"duration {duration:.3f}")
        lines.append(f"file '{frame_paths[-1].as_posix()}'")
        concat.write_text("\n".join(lines) + "\n", encoding="utf-8")

        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat),
            "-vf",
            "fps=1,format=yuv420p",
            "-movflags",
            "+faststart",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        run = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if run.returncode != 0:
            raise RuntimeError("ffmpeg failed: " + sanitize_text(run.stderr, limit=800))


def write_evidence(evidence: Dict[str, object], path: Path = DEFAULT_EVIDENCE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--poster", type=Path, default=DEFAULT_POSTER)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--duration", type=float, default=180.0)
    parser.add_argument(
        "--allow-missing-go",
        action="store_true",
        help="render the primary C->Rust proof even if the C->Go teaser compiler is absent",
    )
    args = parser.parse_args(argv)

    evidence = build_evidence(require_go_teaser=not args.allow_missing_go)
    evidence["duration_seconds"] = args.duration
    write_evidence(evidence, args.evidence)
    render_video(evidence, args.output, args.poster, total_seconds=args.duration)
    print(f"wrote {args.output.relative_to(REPO_ROOT)}")
    print(f"wrote {args.poster.relative_to(REPO_ROOT)}")
    print(f"wrote {args.evidence.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
