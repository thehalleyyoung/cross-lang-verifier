from __future__ import annotations

import importlib.util
import shutil
import subprocess

import pytest

from scripts import build_demo_video as demo


def _sample_evidence():
    return {
        "schema": demo.VIDEO_SCHEMA,
        "primary": {
            "item_id": "nginx-rate",
            "source_library": "nginx",
            "source_function": "nginx_rate",
            "c_file": "experiments/c2rust_corpus/sources/nginx_rate.c",
            "rust_file": "experiments/c2rust_corpus/generated/nginx_rate.rs",
            "translator": "c2rust",
            "translator_version": "C2Rust 0.22.1",
            "c2rust_hallmarks": True,
            "symbolic_verdict": "candidate",
            "symbolic_detail": "symbolic witness for div_by_zero",
            "confirmation": {
                "confirmed": True,
                "reason": "UB reachable; Rust defined",
                "c_runs": {
                    "san": {
                        "returncode": 1,
                        "stderr": "<tmp>: runtime error: division by zero",
                        "ub_category": "division by zero",
                    }
                },
                "target_run": {
                    "returncode": 101,
                    "stderr": "thread 'main' panicked at division by zero",
                },
            },
        },
        "teaser": {
            "available": True,
            "pair": "c->go",
            "case_id": "div-by-zero",
            "cwe": "CWE-369",
            "title": "Division by zero",
            "confirmation": {
                "target_run": {
                    "returncode": 2,
                    "stderr": "panic: runtime error: integer divide by zero",
                }
            },
        },
    }


def _has_primary_toolchain() -> bool:
    status = demo.toolchain_available()
    return status.full_for("rust") and status.full_for("go")


def test_sanitize_text_removes_paths_and_keeps_ubsan_category():
    text = "\x1b[31m/private/var/folders/demo/c_san.c:7: runtime error: division by zero\x1b[0m"

    sanitized = demo.sanitize_text(text)

    assert "\x1b[" not in sanitized
    assert "<tmp>" in sanitized
    assert "division by zero" in sanitized


def test_story_uses_cwe_class_language_without_claiming_real_cve():
    flat = "\n".join(text for frame in demo.build_story(_sample_evidence()) for _, text in frame)

    assert "CWE-369-class" in flat
    assert "assigned nginx CVE" in flat
    assert "CVE-202" not in flat


@pytest.mark.skipif(not _has_primary_toolchain(), reason="needs clang/UBSan, rustc, and go")
def test_build_evidence_confirms_c2rust_cwe369_and_go_teaser():
    evidence = demo.build_evidence()

    assert evidence["schema"] == demo.VIDEO_SCHEMA
    assert evidence["primary"]["item_id"] == "nginx-rate"
    assert evidence["primary"]["cwe"] == "CWE-369"
    assert evidence["primary"]["translator"] == "c2rust"
    assert evidence["primary"]["c2rust_hallmarks"] is True
    assert evidence["primary"]["symbolic_verdict"] == "candidate"
    assert evidence["primary"]["confirmation"]["confirmed"] is True
    assert evidence["primary"]["confirmation"]["ub_reachable"] is True
    assert evidence["primary"]["confirmation"]["target_defined"] is True
    assert evidence["teaser"]["pair"] == "c->go"
    assert evidence["teaser"]["confirmation"]["confirmed"] is True


@pytest.mark.skipif(importlib.util.find_spec("PIL") is None, reason="Pillow not installed")
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe not installed")
def test_render_video_writes_mp4_and_poster(tmp_path):
    out = tmp_path / "demo.mp4"
    poster = tmp_path / "poster.png"

    demo.render_video(_sample_evidence(), out, poster, total_seconds=3)

    assert out.exists()
    assert poster.exists()
    assert out.stat().st_size > 0
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=format_name", "-of", "default=nw=1", str(out)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert probe.returncode == 0
    assert "format_name=" in probe.stdout
