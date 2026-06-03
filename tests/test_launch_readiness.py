from __future__ import annotations

from pathlib import Path

from scripts.validate_launch_readiness import REPO_ROOT, validate


def test_launch_readiness_runs_real_demo_and_checks_public_packet():
    evidence = validate(REPO_ROOT)

    assert evidence["ok"] is True
    assert "docs/launch.md" in evidence["files"]
    assert "CONTRIBUTING.md" in evidence["files"]
    assert "clang" in evidence["tool_versions"]
    assert "rustc" in evidence["tool_versions"]
    stdout = evidence["demo"]["stdout"]
    assert "DIVERGENT" in stdout
    assert "UB reachable" in stdout
    assert "NO-DIVERGENCE" in stdout
    assert "abstract-interpretation" in stdout
    assert "signed_overflow" in evidence["demo"]["sarif_rule_ids"]


def test_launch_packet_mentions_only_checked_local_assets():
    launch = (Path(REPO_ROOT) / "docs/launch.md").read_text(encoding="utf-8")

    assert "python3 scripts/validate_launch_readiness.py" in launch
    assert "External posting" in launch or "external posting" in launch
    assert "docs/assets/demo_video.mp4" in launch
    assert "examples/plugins/float_cast_overflow_oracle.py" in launch
