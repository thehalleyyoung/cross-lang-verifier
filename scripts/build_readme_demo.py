#!/usr/bin/env python3
"""Regenerate the README's animated verifier demo from the real CLI.

The generated GIF is intentionally a checked-in artifact: repository users can
see the demo without installing image tooling, while maintainers can refresh it
with a real verifier run when claims or output wording change.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path("examples/readme_demo_units.json")
OUTPUT = Path("docs/assets/readme_demo.gif")
DISPLAY_COMMAND = (
    "python3 -m src.ub_oracle.cli --units examples/readme_demo_units.json "
    "--format text --color never --fail-on unknown"
)

StyledLine = Tuple[str, str]


def demo_command(root: Path = REPO_ROOT) -> List[str]:
    """Return the exact command used to capture the demo output."""
    return [
        sys.executable,
        "-m",
        "src.ub_oracle.cli",
        "--units",
        str(root / MANIFEST),
        "--format",
        "text",
        "--color",
        "never",
        "--fail-on",
        "unknown",
    ]


def run_demo(root: Path = REPO_ROOT, *, allow_symbolic: bool = False) -> str:
    """Run the real verifier and return stdout.

    By default, the README demo must show a confirmed divergence, which requires
    clang/UBSan and rustc on PATH. ``--allow-symbolic`` lets maintainers inspect
    the animation path on hosts without those compilers, but it will not refresh
    the checked-in hero asset by accident.
    """
    proc = subprocess.run(
        demo_command(root),
        cwd=root,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "README demo command failed:\n"
            f"$ {DISPLAY_COMMAND}\n\nstdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
    output = sanitize_output(proc.stdout)
    confirmed = "DIVERGENT" in output and "UB reachable" in output
    if not confirmed and not allow_symbolic:
        raise RuntimeError(
            "README demo requires a confirmed real-compiler divergence. "
            "Install clang with UBSan support and rustc, or rerun with "
            "--allow-symbolic for local inspection only."
        )
    return output


def sanitize_output(output: str) -> str:
    """Normalize environment-dependent text without changing the verdict."""
    output = re.sub(r"\x1b\[[0-9;]*m", "", output)
    output = re.sub(r"/[^\s:]*tmp[^\s:]*", "<tmp>", output)
    output = re.sub(r"\s+\n", "\n", output)
    return output.strip()


def _styled(text: str, style: str = "text") -> StyledLine:
    return (style, text)


def _style_for_output(line: str) -> str:
    if "DIVERGENT" in line:
        return "danger"
    if "NO-DIVERGENCE" in line:
        return "ok"
    if "Summary" in line or line.startswith("cross-lang-verify"):
        return "accent"
    if "UB reachable" in line or "abstract-interpretation" in line:
        return "dim"
    return "text"


def _keep_cli_lines(output: str) -> List[str]:
    """Select the high-signal lines for the animation."""
    selected: List[str] = []
    for line in output.splitlines():
        stripped = line.rstrip()
        if not stripped:
            continue
        if (
            stripped.startswith("cross-lang-verify")
            or "DIVERGENT" in stripped
            or "NO-DIVERGENCE" in stripped
            or "UB reachable" in stripped
            or "abstract-interpretation" in stripped
            or stripped.startswith("Summary")
            or stripped.lstrip().startswith("decided")
            or stripped.startswith("  NO_DIVERGENCE_FOUND")
        ):
            selected.append(stripped)
    return selected


def build_story(output: str) -> List[List[StyledLine]]:
    """Turn verifier output into a small sequence of terminal frames."""
    cli = _keep_cli_lines(output)
    intro = [
        _styled("cross-lang-verifier", "accent"),
        _styled("C -> Rust translation validation grounded in real compilers.", "dim"),
        _styled(""),
        _styled("C source:", "prompt"),
        _styled("int f(int x) { return x + 1; }"),
        _styled(""),
        _styled("Rust port:", "prompt"),
        _styled("pub fn f(x: i32) -> i32 { x.wrapping_add(1) }"),
    ]
    command = intro + [
        _styled(""),
        _styled("$ " + DISPLAY_COMMAND, "command"),
    ]
    first_result = command + [_styled("")]
    first_result.extend(_styled(line, _style_for_output(line)) for line in cli[:3])
    full_result = command + [_styled("")]
    full_result.extend(_styled(line, _style_for_output(line)) for line in cli[:7])
    summary = command + [_styled("")]
    summary.extend(_styled(line, _style_for_output(line)) for line in cli)
    return [intro, command, first_result, full_result, summary]


def render_gif(
    frames: Sequence[Sequence[StyledLine]],
    destination: Path,
    *,
    size: Tuple[int, int] = (1180, 640),
    duration_ms: int = 1450,
) -> None:
    """Render styled terminal frames to an animated GIF."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:  # pragma: no cover - depends on optional tooling.
        raise RuntimeError(
            "Pillow is required to render docs/assets/readme_demo.gif. "
            "Install it with: python3 -m pip install Pillow"
        ) from exc

    font = _load_font(ImageFont)
    images = [_render_frame(Image, ImageDraw, font, frame, size) for frame in frames]
    destination.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        destination,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )


def _load_font(image_font_module):
    candidates = [
        "/System/Library/Fonts/Menlo.ttc",
        "/Library/Fonts/Menlo.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return image_font_module.truetype(candidate, 24)
    return image_font_module.load_default()


def _render_frame(image_module, draw_module, font, lines: Sequence[StyledLine], size):
    width, height = size
    image = image_module.new("RGB", size, "#07111f")
    draw = draw_module.Draw(image)
    draw.rounded_rectangle((24, 24, width - 24, height - 24), radius=18, fill="#0f172a")
    draw.rounded_rectangle((24, 24, width - 24, height - 24), radius=18, outline="#334155", width=2)
    for i, color in enumerate(("#ef4444", "#f59e0b", "#22c55e")):
        draw.ellipse((50 + i * 28, 48, 68 + i * 28, 66), fill=color)
    draw.text((150, 44), "confirmed C -> Rust divergence", fill="#93c5fd", font=font)

    palette = {
        "accent": "#60a5fa",
        "command": "#fbbf24",
        "danger": "#fb7185",
        "dim": "#94a3b8",
        "ok": "#34d399",
        "prompt": "#a7f3d0",
        "text": "#e5e7eb",
    }
    y = 92
    line_height = 34
    for style, text in lines:
        if y > height - 62:
            break
        draw.text((58, y), text, fill=palette.get(style, palette["text"]), font=font)
        y += line_height
    return image


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / OUTPUT)
    parser.add_argument(
        "--allow-symbolic",
        action="store_true",
        help="allow a symbolic candidate-only run if the real compilers are absent",
    )
    args = parser.parse_args(argv)

    output = run_demo(REPO_ROOT, allow_symbolic=args.allow_symbolic)
    frames = build_story(output)
    render_gif(frames, args.output)
    print(f"wrote {args.output.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
