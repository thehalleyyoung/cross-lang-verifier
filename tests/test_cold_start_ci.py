from __future__ import annotations

import os
import subprocess
from types import SimpleNamespace

from scripts import cold_start_ci as cold
from src.ub_oracle.reexec import NO_TOOLCHAIN_ENV, toolchain_available


def test_toolchain_mask_makes_discovery_deterministically_unavailable(monkeypatch):
    monkeypatch.setenv(NO_TOOLCHAIN_ENV, "1")

    status = toolchain_available()

    assert status.cc is None
    assert not status.ubsan
    assert not status.full_for("rust")
    assert all(path is None for _, path in status.targets)


def test_noncompiler_pytest_uses_toolchain_mask_and_parallel_flag(monkeypatch):
    monkeypatch.setattr(cold, "_xdist_available", lambda: True)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout="123 passed", stderr="")

    phase = cold.run_noncompiler_pytest(timeout_seconds=5, runner=fake_run)

    assert phase.returncode == 0
    cmd, kwargs = calls[0]
    assert cmd[:3] == [os.sys.executable, "-m", "pytest"]
    assert "-n" in cmd and "auto" in cmd
    assert kwargs["env"][NO_TOOLCHAIN_ENV] == "1"
    assert "123 passed" in phase.detail


def test_noncompiler_pytest_reports_timeout():
    def fake_run(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(["pytest"], timeout=1, output="partial", stderr="slow")

    phase = cold.run_noncompiler_pytest(timeout_seconds=1, runner=fake_run)

    assert phase.returncode == 124
    assert "timed out" in phase.detail
    assert "partial" in phase.detail


def test_gate_rejects_budget_exhaustion_before_sample(monkeypatch):
    monkeypatch.setattr(
        cold,
        "run_noncompiler_pytest",
        lambda **_kwargs: cold.PhaseResult("noncompiler_pytest", 0.01, 0, "ok"),
    )
    ticks = iter([0.0, 2.0])
    monkeypatch.setattr(cold.time, "monotonic", lambda: next(ticks))

    args = SimpleNamespace(
        budget_seconds=1.0,
        sample_reserve_seconds=0.1,
        sample_timeout_seconds=1,
        no_parallel=True,
        skip_pytest=False,
        skip_sample=False,
        pytest_arg=[],
    )

    assert cold.run_gate(args) == 124
