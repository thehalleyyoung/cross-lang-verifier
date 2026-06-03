from __future__ import annotations

import importlib.util

import pytest

from scripts.build_readme_demo import build_story, render_gif, sanitize_output


SAMPLE_OUTPUT = """\x1b[31mcross-lang-verify\x1b[0m  (2 unit(s); toolchain available)

  DIVERGENT               c_rust_signed_overflow  [c->rust]
      UB reachable (sanitizer trapped); O0='0' vs O2='1' differ; Rust defined='0'
  NO-DIVERGENCE           c_rust_safe_add_zero  [c->rust]
      checked classes: [signed_overflow]; 1 class(es) discharged by abstract-interpretation pre-pass without SMT

Summary — 2 unit(s)
  decided   : 2  (100%)  [divergent=1, no_divergence_found=1]
  NO_DIVERGENCE_FOUND means only the covered divergence classes were checked and none fired; it is NOT a proof of equivalence.
"""


def test_sanitize_output_preserves_verdicts_and_strips_terminal_noise():
    noisy = SAMPLE_OUTPUT + "\n/tmp/readme-demo/c_san.c:3: runtime error: signed overflow\n"

    sanitized = sanitize_output(noisy)

    assert "\x1b[" not in sanitized
    assert "DIVERGENT" in sanitized
    assert "NO-DIVERGENCE" in sanitized
    assert "<tmp>" in sanitized


def test_story_leads_with_real_code_and_checked_verdict():
    frames = build_story(sanitize_output(SAMPLE_OUTPUT))
    flat_text = "\n".join(text for frame in frames for _, text in frame)

    assert "int f(int x) { return x + 1; }" in flat_text
    assert "x.wrapping_add(1)" in flat_text
    assert "UB reachable" in flat_text
    assert "NO_DIVERGENCE_FOUND" in flat_text


@pytest.mark.skipif(importlib.util.find_spec("PIL") is None, reason="Pillow not installed")
def test_render_gif_writes_animated_gif(tmp_path):
    from PIL import Image

    out = tmp_path / "readme_demo.gif"
    render_gif(build_story(sanitize_output(SAMPLE_OUTPUT)), out, duration_ms=20)

    with Image.open(out) as gif:
        assert gif.format == "GIF"
        assert gif.n_frames >= 3
