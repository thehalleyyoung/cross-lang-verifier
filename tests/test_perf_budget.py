from __future__ import annotations

from experiments.perf_curves import run as perf_run
from src.ub_oracle import perf


def test_percentile_uses_conservative_nearest_rank():
    samples = [0.001, 0.002, 0.003, 0.100]

    assert perf.percentile(samples, 50) == 0.002
    assert perf.percentile(samples, 95) == 0.100
    assert perf.percentile(samples, 0) == 0.001
    assert perf.percentile(samples, 100) == 0.100


def test_latency_budget_row_reports_p50_and_p95_failures():
    row = perf.LatencyBudgetRow(
        label="slow-oracle",
        samples=[0.010, 0.011, 0.500],
        verdict="divergent",
        budget=perf.LatencyBudget(p50_seconds=0.020, p95_seconds=0.100),
    )

    assert row.p50_seconds == 0.011
    assert row.p95_seconds == 0.500
    assert not row.ok
    assert row.failures() == ["slow-oracle p95 0.500000s > budget 0.100000s"]
    assert row.to_dict()["budget_p95_seconds"] == 0.100


def test_real_smt_probe_can_be_budgeted():
    row = perf._measure_latency_row(
        "smt-overflow-smoke",
        lambda: perf._solve_overflow_at_width(8),
        repeats=1,
        budget=perf.LatencyBudget(p50_seconds=1.0, p95_seconds=1.0),
    )

    assert row.ok
    assert row.verdict == "sat"
    assert row.samples and row.p95_seconds >= row.p50_seconds


def test_perf_check_returns_nonzero_on_latency_budget_failure(tmp_path, monkeypatch, capsys):
    grid_path = tmp_path / "grid.json"
    grid_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(perf_run, "GRID_PATH", str(grid_path))
    monkeypatch.setattr(perf_run.perf, "deterministic_grid", lambda: {})
    monkeypatch.setattr(
        perf_run.perf,
        "latency_budget_report",
        lambda repeats: {
            "schema": "perf-budget-report/v1",
            "repeats": repeats,
            "budgets": {},
            "class_pair_profile": [{
                "label": "slow-oracle",
                "p95_seconds": 2.0,
                "ok": False,
            }],
            "width_scaling": [],
            "smt_scaling": [],
            "ok": False,
            "failures": ["slow-oracle p95 2.000000s > budget 1.000000s"],
        },
    )

    assert perf_run.main(["--check", "--budget-check"]) == 1
    assert "slow-oracle p95" in capsys.readouterr().out
